// Sokoban, solved room by room with IDA* over pushes.
//
// Every level this contest draws is a hallway with small rooms off it.
// A box can never leave its room (a push out of the door cannot be undone),
// so each room is an independent push-optimal problem: the player walks in
// from the hallway square outside the door, solves the room, and walks out.
// Inside a room the search is IDA* over pushes, guided by a pattern database
// (exact one-box push distances to every goal, combined by minimum-cost
// matching) and pruned by a transposition table. All budgets are counted in
// expanded nodes, never in time, so two runs write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Steps in reading order: up, down, left, right. DY/DX pair with CH.
const DY = [-1, 1, 0, 0];
const DX = [0, 0, -1, 1];
const CH = ['u', 'd', 'l', 'r'];

// A parsed level: flat cell arrays over the whole board.
function parseLevel(text) {
  const rows = text.replace(/\r/g, '').split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < rows[y].length; x++) {
      const c = rows[y][x];
      const i = y * w + x;
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') goal[i] = 1;
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// The rooms of a level, each re-based onto the generator's room board:
// 8 wide, rows 0..9, the slot in rows 1..6 and columns 1..6, the door in
// row 7, the hallway square in row 8. A room above the hallway reads the
// board as drawn; a room below reads it flipped top to bottom. Local row 8
// is masked to the door column and row 9 is wall: the search may step out
// of the door onto the hallway square, never wander the hallway itself.
const ROOM_W = 8;
const SLOT_PITCH = 7;

function extractRooms(L) {
  const hallY = Math.floor(L.player / L.w);
  const slots = (L.w - 1) / SLOT_PITCH;
  const rooms = [];
  for (let s = 0; s < slots; s++) {
    for (const below of [false, true]) {
      if (below && L.h <= 10) continue; // a 10-row board holds rooms above only
      const toFull = (lx, ly) => {
        const fx = SLOT_PITCH * s + lx;
        const fy = below ? L.h - 1 - ly : ly;
        return fy * L.w + fx;
      };
      // A room is present when its slot holds a box or a goal.
      const boxes = [];
      const goals = [];
      for (let ly = 1; ly <= 7; ly++) {
        for (let lx = 1; lx <= 6; lx++) {
          const f = toFull(lx, ly);
          if (L.goal[f]) goals.push(ly * ROOM_W + lx);
          if (L.boxes.includes(f)) boxes.push(ly * ROOM_W + lx);
        }
      }
      if (boxes.length === 0) continue;
      const wall = new Uint8Array(ROOM_W * 10).fill(1);
      for (let ly = 0; ly <= 7; ly++) {
        for (let lx = 0; lx < ROOM_W; lx++) {
          if (!L.wall[toFull(lx, ly)]) wall[ly * ROOM_W + lx] = 0;
        }
      }
      // The door is the one open cell of row 7; row 8 opens only there.
      let door = -1;
      for (let lx = 1; lx <= 6; lx++) if (!wall[7 * ROOM_W + lx]) door = lx;
      if (door !== -1) wall[8 * ROOM_W + door] = 0;
      boxes.sort((a, b) => a - b);
      goals.sort((a, b) => a - b);
      rooms.push({
        wall, goals, boxes, door, below, slot: s,
        start: 8 * ROOM_W + door,
        toFull,
        hallFull: toFull(door, 8),
      });
    }
  }
  return { rooms, hallY };
}

// The cell reached from i by one step in direction d on a board of width w
// holding n cells, or -1 when the step leaves the board.
function stepCell(i, d, w, n) {
  const x = i % w;
  if (d === 2 && x === 0) return -1;
  if (d === 3 && x === w - 1) return -1;
  const j = i + DY[d] * w + DX[d];
  return j < 0 || j >= n ? -1 : j;
}

// The pattern database: for each goal, the exact number of pushes that
// brings a lone box from any cell onto it, computed backwards from the goal
// (a box at c was pushed there from c - d, with the player standing at
// c - 2d; both cells must be floor). Other boxes are ignored, so the table
// never overstates: it is a true lower bound for one box.
function pushTables(wall, goals, w) {
  const n = wall.length;
  return goals.map((g) => {
    const dist = new Int16Array(n).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi];
      for (let d = 0; d < 4; d++) {
        const back = d ^ 1;
        const from = stepCell(c, back, w, n);
        if (from < 0 || wall[from]) continue;
        const stand = stepCell(from, back, w, n);
        if (stand < 0 || wall[stand] || dist[from] !== -1) continue;
        dist[from] = dist[c] + 1;
        queue.push(from);
      }
    }
    return dist;
  });
}

function permutations(k) {
  if (k <= 1) return [[0].slice(0, k)];
  const out = [];
  for (const rest of permutations(k - 1)) {
    for (let i = 0; i < k; i++) {
      const p = rest.slice();
      p.splice(i, 0, k - 1);
      out.push(p);
    }
  }
  return out;
}

// The heuristic: the cheapest assignment of boxes to goals under the
// pattern database (a minimum-cost matching, by permutation: at most four
// boxes). A box that reaches no goal makes the state dead: Infinity.
function makeHeuristic(tables, nBoxes) {
  const perms = permutations(nBoxes);
  return (boxes) => {
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < boxes.length; i++) {
        const d = tables[perm[i]][boxes[i]];
        if (d < 0) { sum = Infinity; break; }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    return best;
  };
}

// Every cell the player reaches from `from` without crossing a box, with
// the step that first reached each cell, so a walk can be rebuilt.
function flood(wall, occupied, from, w) {
  const parent = new Int16Array(wall.length).fill(-1);
  const via = new Int8Array(wall.length).fill(-1);
  parent[from] = from;
  const queue = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    for (let d = 0; d < 4; d++) {
      const j = stepCell(c, d, w, wall.length);
      if (j < 0 || wall[j] || occupied[j] || parent[j] !== -1) continue;
      parent[j] = c;
      via[j] = d;
      queue.push(j);
    }
  }
  return { parent, via };
}

// The moves of the walk flood() found between two cells of one region.
function walkString(f, from, to) {
  let s = '';
  for (let c = to; c !== from; c = f.parent[c]) s = CH[f.via[c]] + s;
  return s;
}

// IDA* over pushes for one room. A node is the sorted box cells plus the
// player's cell just after a push; the transposition table keys a node by
// its boxes and the smallest cell of the player's reachable region, and
// prunes a revisit that arrives with no fewer pushes. The bound starts at
// the heuristic of the start and rises to the smallest f that exceeded it,
// so the first solution found is push-optimal. Should IDA* ever spend its
// node budget (a budget of work, never of time), a plain A* over the same
// heuristic finishes the room, still push-optimally.
const IDA_NODE_CAP = 3000000;
const ASTAR_NODE_CAP = 4000000;

function solveRoom(room) {
  const { wall, goals, boxes, start } = room;
  const w = ROOM_W;
  const n = wall.length;
  const tables = pushTables(wall, goals, w);
  const heuristic = makeHeuristic(tables, boxes.length);
  const onGoal = (b) => goals.includes(b);
  if (heuristic(boxes) === Infinity) return null;
  if (boxes.every(onGoal)) return { moves: '', player: start, boxes };

  const table = new Map();
  const path = [];
  let nodes = 0;
  let aborted = false;
  let endPlayer = -1;
  let endBoxes = null;

  const search = (cur, player, g, bound) => {
    const f = g + heuristic(cur);
    if (f > bound) return { found: false, next: f };
    if (cur.every(onGoal)) {
      endPlayer = player;
      endBoxes = cur;
      return { found: true, next: Infinity };
    }
    const occupied = new Uint8Array(n);
    for (const b of cur) occupied[b] = 1;
    const reach = flood(wall, occupied, player, w);
    let rep = player;
    for (let c = 0; c < n; c++) if (reach.parent[c] !== -1 && c < rep) rep = c;
    const key = cur.join(',') + '|' + rep;
    const seen = table.get(key);
    if (seen !== undefined && seen <= g) return { found: false, next: Infinity };
    table.set(key, g);
    nodes++;
    if (nodes > IDA_NODE_CAP) {
      aborted = true;
      return { found: false, next: Infinity };
    }
    let next = Infinity;
    for (let bi = 0; bi < cur.length; bi++) {
      const b = cur[bi];
      for (let d = 0; d < 4; d++) {
        const stand = stepCell(b, d ^ 1, w, n);
        if (stand < 0 || reach.parent[stand] === -1) continue;
        const dest = stepCell(b, d, w, n);
        if (dest < 0 || wall[dest] || occupied[dest]) continue;
        const moved = cur.slice();
        moved[bi] = dest;
        moved.sort((a, b2) => a - b2);
        path.push(walkString(reach, player, stand) + CH[d]);
        const r = search(moved, b, g + 1, bound);
        if (r.found) return r;
        path.pop();
        if (aborted) return { found: false, next: Infinity };
        if (r.next < next) next = r.next;
      }
    }
    return { found: false, next };
  };

  let bound = heuristic(boxes);
  for (;;) {
    const r = search(boxes, start, 0, bound);
    if (r.found) return { moves: path.join(''), player: endPlayer, boxes: endBoxes };
    if (aborted || r.next === Infinity) break;
    bound = r.next;
  }
  return astarRoom(room, tables, heuristic);
}

// The A* fallback: same states, same heuristic, a closed map in place of
// the transposition table. Deterministic: ties break by deeper node, then
// by insertion order.
function astarRoom(room, tables, heuristic) {
  const { wall, goals, boxes, start } = room;
  const w = ROOM_W;
  const n = wall.length;
  const onGoal = (b) => goals.includes(b);
  const closed = new Map();
  const heap = [];
  let order = 0;
  const push = (node) => {
    heap.push(node);
    for (let i = heap.length - 1; i > 0;) {
      const p = (i - 1) >> 1;
      if (!before(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && before(heap[l], heap[m])) m = l;
        if (r < heap.length && before(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };
  const before = (a, b) => a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.order < b.order)));
  push({ f: heuristic(boxes), g: 0, boxes, player: start, moves: '', order: order++ });
  let nodes = 0;
  while (heap.length > 0) {
    const node = pop();
    if (node.boxes.every(onGoal)) return { moves: node.moves, player: node.player, boxes: node.boxes };
    const occupied = new Uint8Array(n);
    for (const b of node.boxes) occupied[b] = 1;
    const reach = flood(wall, occupied, node.player, w);
    let rep = node.player;
    for (let c = 0; c < n; c++) if (reach.parent[c] !== -1 && c < rep) rep = c;
    const key = node.boxes.join(',') + '|' + rep;
    const seen = closed.get(key);
    if (seen !== undefined && seen <= node.g) continue;
    closed.set(key, node.g);
    if (++nodes > ASTAR_NODE_CAP) return null;
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      for (let d = 0; d < 4; d++) {
        const stand = stepCell(b, d ^ 1, w, n);
        if (stand < 0 || reach.parent[stand] === -1) continue;
        const dest = stepCell(b, d, w, n);
        if (dest < 0 || wall[dest] || occupied[dest]) continue;
        const moved = node.boxes.slice();
        moved[bi] = dest;
        moved.sort((a, b2) => a - b2);
        const g = node.g + 1;
        const h = heuristic(moved);
        if (h === Infinity) continue;
        push({
          f: g + h, g, boxes: moved, player: b,
          moves: node.moves + walkString(reach, node.player, stand) + CH[d],
          order: order++,
        });
      }
    }
  }
  return null;
}

// A room below the hallway is solved on a board flipped top to bottom, so
// its up and down swap when its moves are played on the real board.
function flipMoves(s, below) {
  if (!below) return s;
  let out = '';
  for (const c of s) out += c === 'u' ? 'd' : c === 'd' ? 'u' : c;
  return out;
}

function solve(levelText) {
  const L = parseLevel(levelText);
  const { rooms } = extractRooms(L);
  if (rooms.length === 0) return '';
  const boxSet = new Set(L.boxes);
  let player = L.player;
  let out = '';
  for (const room of rooms) {
    const solved = solveRoom(room);
    if (solved === null) return '';
    // Walk the board to the hallway square outside this room's door.
    let occupied = new Uint8Array(L.wall.length);
    for (const b of boxSet) occupied[b] = 1;
    const along = flood(L.wall, occupied, player, L.w);
    if (along.parent[room.hallFull] === -1) return '';
    out += walkString(along, player, room.hallFull);
    player = room.hallFull;
    // Play the room's solution on the real board.
    out += flipMoves(solved.moves, room.below);
    for (const b of room.boxes) boxSet.delete(room.toFull(b % ROOM_W, Math.floor(b / ROOM_W)));
    for (const b of solved.boxes) boxSet.add(room.toFull(b % ROOM_W, Math.floor(b / ROOM_W)));
    player = room.toFull(solved.player % ROOM_W, Math.floor(solved.player / ROOM_W));
    // Walk out of the finished room, back to its hallway square.
    occupied = new Uint8Array(room.wall.length);
    for (const b of solved.boxes) occupied[b] = 1;
    const exit = flood(room.wall, occupied, solved.player, ROOM_W);
    if (exit.parent[room.start] === -1) return '';
    out += flipMoves(walkString(exit, solved.player, room.start), room.below);
    player = room.hallFull;
  }
  return out;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
