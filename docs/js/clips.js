// Labelled clips saved from the live microphone, kept in this browser (IndexedDB).
// Falls back to memory (lost on reload) where IndexedDB is unavailable, e.g. some private windows.
const DB_NAME = 'shotshield-clips';
const META = 'meta';
const AUDIO = 'audio';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(META, { keyPath: 'id', autoIncrement: true });
      req.result.createObjectStore(AUDIO);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function done(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export class ClipStore {
  static async open() {
    const s = new ClipStore();
    try {
      s.db = await openDb();
    } catch (err) {
      console.warn('IndexedDB unavailable, saved clips will be lost on reload', err);
      s.db = null;
      s.mem = new Map();
      s.nextId = 1;
    }
    return s;
  }

  get persistent() { return this.db !== null; }

  async add(meta, wav) {
    if (!this.db) {
      const id = this.nextId++;
      this.mem.set(id, { meta: { ...meta, id }, wav });
      return id;
    }
    const tx = this.db.transaction([META, AUDIO], 'readwrite');
    let id;
    const req = tx.objectStore(META).add(meta);
    req.onsuccess = () => { id = req.result; tx.objectStore(AUDIO).put(wav, id); };
    await done(tx);
    return id;
  }

  async list() {
    if (!this.db) return [...this.mem.values()].map((x) => x.meta);
    const tx = this.db.transaction(META, 'readonly');
    const req = tx.objectStore(META).getAll();
    await done(tx);
    return req.result;
  }

  async audio(id) {
    if (!this.db) return this.mem.get(id)?.wav;
    const tx = this.db.transaction(AUDIO, 'readonly');
    const req = tx.objectStore(AUDIO).get(id);
    await done(tx);
    return req.result;
  }

  async update(id, patch) {
    if (!this.db) { Object.assign(this.mem.get(id).meta, patch); return; }
    const tx = this.db.transaction(META, 'readwrite');
    const store = tx.objectStore(META);
    const req = store.get(id);
    req.onsuccess = () => store.put({ ...req.result, ...patch });
    await done(tx);
  }

  async remove(id) {
    if (!this.db) { this.mem.delete(id); return; }
    const tx = this.db.transaction([META, AUDIO], 'readwrite');
    tx.objectStore(META).delete(id);
    tx.objectStore(AUDIO).delete(id);
    await done(tx);
  }

  async clear() {
    if (!this.db) { this.mem.clear(); return; }
    const tx = this.db.transaction([META, AUDIO], 'readwrite');
    tx.objectStore(META).clear();
    tx.objectStore(AUDIO).clear();
    await done(tx);
  }
}
