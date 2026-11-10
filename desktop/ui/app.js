// Laras desktop — antarmuka. Tanpa bundler, tanpa npm: `withGlobalTauri`
// menyuntikkan `window.__TAURI__` sehingga berkas ini cukup dimuat apa adanya.

const { invoke } = window.__TAURI__.core;
const { convertFileSrc } = window.__TAURI__.core;

const $ = (id) => document.getElementById(id);

const el = {
  pil: $('status-pil'),
  statusTeks: $('status-teks'),
  panel: $('panel-setup'),
  tombolSetup: $('tombol-setup'),
  models: $('jalur-models'),
  suara: $('jalur-suara'),
  pilihModels: $('pilih-models'),
  pilihSuara: $('pilih-suara'),
  muat: $('tombol-muat'),
  infoCpu: $('info-cpu'),
  teks: $('teks'),
  buat: $('tombol-buat'),
  hitung: $('hitung'),
  pesan: $('pesan'),
  hasil: $('hasil'),
  pemutar: $('pemutar'),
  metaDurasi: $('meta-durasi'),
  metaWaktu: $('meta-waktu'),
  simpan: $('tombol-simpan'),
  folder: $('tombol-folder'),
  versi: $('versi'),
};

let berkasTerakhir = null;
let siap = false;

// --------------------------------------------------------------- pembantu

function tampilkanPesan(teks, jenis = 'galat') {
  el.pesan.textContent = teks;
  el.pesan.className = 'pesan' + (jenis === 'info' ? ' info' : '');
  el.pesan.hidden = false;
}

function sembunyikanPesan() {
  el.pesan.hidden = true;
}

function aturStatus(teks, kelas = '') {
  el.statusTeks.textContent = teks;
  el.pil.className = 'pil' + (kelas ? ' ' + kelas : '');
}

function sibuk(tombol, teksSibuk) {
  if (!tombol.dataset.asli) tombol.dataset.asli = tombol.textContent;
  tombol.disabled = true;
  tombol.innerHTML = '<span class="berputar"></span> ' + teksSibuk;
}

function selesaiSibuk(tombol) {
  tombol.disabled = false;
  tombol.textContent = tombol.dataset.asli || 'Buat suara';
}

function perbaruiHitung() {
  const n = el.teks.value.length;
  el.hitung.textContent = n + ' karakter';
}

// ----------------------------------------------------------------- status

async function segarkan(s) {
  siap = s.siap;
  el.models.value = s.models;
  el.suara.value = s.referensi;
  el.infoCpu.textContent =
    s.inti_cpu + ' inti CPU terdeteksi · versi ' + s.versi;
  el.versi.textContent = 'Laras ' + s.versi;

  if (s.siap) {
    aturStatus('siap', 'siap');
    el.panel.hidden = true;
    el.muat.textContent = 'Muat ulang mesin';
  } else {
    aturStatus('model belum dimuat');
    el.panel.hidden = false;
    el.muat.textContent = 'Muat mesin';
  }
  el.buat.disabled = !s.siap;
}

async function muatStatus() {
  try {
    await segarkan(await invoke('status'));
  } catch (e) {
    aturStatus('galat', 'galat');
    tampilkanPesan('Tidak bisa membaca status: ' + e);
  }
}

// ------------------------------------------------------------------ muat

el.muat.addEventListener('click', async () => {
  sembunyikanPesan();
  sibuk(el.muat, 'memuat model…');
  aturStatus('memuat model…');
  try {
    const s = await invoke('siapkan', {
      models: el.models.value.trim() || null,
      referensi: el.suara.value.trim() || null,
    });
    await segarkan(s);
    tampilkanPesan(
      'Mesin siap. Model dimuat dari ' + s.models + ' — sekarang bisa sintesis tanpa jaringan.',
      'info'
    );
  } catch (e) {
    aturStatus('model belum dimuat', 'galat');
    tampilkanPesan(String(e));
    el.panel.hidden = false;
  } finally {
    el.muat.disabled = false;
    el.muat.textContent = siap ? 'Muat ulang mesin' : 'Muat mesin';
  }
});

el.pilihModels.addEventListener('click', async () => {
  const jalur = await invoke('pilih_folder');
  if (jalur) el.models.value = jalur;
});

el.pilihSuara.addEventListener('click', async () => {
  const jalur = await invoke('pilih_suara');
  if (jalur) el.suara.value = jalur;
});

el.tombolSetup.addEventListener('click', () => {
  el.panel.hidden = !el.panel.hidden;
});

// --------------------------------------------------------------- sintesis

async function buat() {
  if (!siap) {
    tampilkanPesan('Muat mesin dulu lewat tombol “Pengaturan”.');
    el.panel.hidden = false;
    return;
  }
  const teks = el.teks.value.trim();
  if (!teks) {
    tampilkanPesan('Teksnya masih kosong.');
    return;
  }

  sembunyikanPesan();
  sibuk(el.buat, 'menyintesis…');
  aturStatus('menyintesis…');
  const mulai = performance.now();

  try {
    const hasil = await invoke('sintesis', { teks });
    const total = ((performance.now() - mulai) / 1000).toFixed(2);

    berkasTerakhir = hasil.berkas;
    el.pemutar.src = convertFileSrc(hasil.berkas);
    el.pemutar.load();
    el.hasil.hidden = false;

    el.metaDurasi.textContent =
      hasil.durasi.toFixed(2) + ' s audio · ' +
      Math.round(hasil.byte / 1024) + ' KB';
    el.metaWaktu.textContent = total + ' s proses (take ' + (hasil.take + 1) + ')';

    aturStatus('siap', 'siap');
    el.pemutar.play().catch(() => {});
  } catch (e) {
    aturStatus('siap', 'siap');
    tampilkanPesan(String(e));
  } finally {
    selesaiSibuk(el.buat);
  }
}

el.buat.addEventListener('click', buat);

el.teks.addEventListener('input', perbaruiHitung);
el.teks.addEventListener('keydown', (ev) => {
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
    ev.preventDefault();
    buat();
  }
});

// ------------------------------------------------------------ simpan/folder

el.simpan.addEventListener('click', async () => {
  if (!berkasTerakhir) return;
  try {
    const tujuan = await invoke('simpan_wav', { sumber: berkasTerakhir });
    if (tujuan) tampilkanPesan('Disimpan ke ' + tujuan, 'info');
  } catch (e) {
    tampilkanPesan(String(e));
  }
});

el.folder.addEventListener('click', () => {
  if (berkasTerakhir) invoke('buka_folder', { jalur: berkasTerakhir });
});

// ------------------------------------------------------------------- awal

perbaruiHitung();
muatStatus();
