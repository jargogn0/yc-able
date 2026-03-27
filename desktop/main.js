const { app, BrowserWindow, shell, Menu } = require('electron');
const path = require('path');

// Must be called before app is ready
app.disableHardwareAcceleration();

const APP_URL = process.env.LABS_URL || 'https://yc-able.com/app';

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#09090b',
    show: false,   // hidden until ready-to-show fires (after first real paint)
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
      allowRunningInsecureContent: true,
    },
  });

  // Splash overlay
  const splash = new BrowserWindow({
    width: 380,
    height: 280,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
  });
  splash.loadFile(path.join(__dirname, 'splash.html'));
  splash.center();

  function closeSplash() {
    try { if (splash && !splash.isDestroyed()) splash.destroy(); } catch (_) {}
  }

  function showMain() {
    closeSplash();
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show();
      mainWindow.focus();
    }
  }

  // ready-to-show fires after the FIRST PAINT — content is actually visible
  mainWindow.once('ready-to-show', () => {
    showMain();
  });

  // Safety net: always show after 12s even if ready-to-show never fires
  const safetyTimer = setTimeout(showMain, 12000);
  mainWindow.once('ready-to-show', () => clearTimeout(safetyTimer));

  mainWindow.loadURL(APP_URL);

  // Retry on network failure
  let retries = 0;
  mainWindow.webContents.on('did-fail-load', (e, code) => {
    if (code === -3) return; // navigation aborted — not an error
    retries++;
    if (retries <= 3) {
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
      }, retries * 2000);
    } else {
      showMain(); // show the (blank) window so user isn't stuck on splash
    }
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    clearTimeout(safetyTimer);
    mainWindow = null;
  });
}

function buildMenu() {
  const isMac = process.platform === 'darwin';
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(isMac ? [{ label: '19Labs', submenu: [
      { role: 'about' },
      { type: 'separator' },
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
      }},
      { type: 'separator' },
      { role: 'services' },
      { type: 'separator' },
      { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' },
      { type: 'separator' },
      { role: 'quit' }
    ]}] : []),
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => mainWindow && mainWindow.loadURL(APP_URL) },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
      { type: 'separator' }, { role: 'togglefullscreen' }
    ]},
    { label: 'Window', submenu: [{ role: 'minimize' }, { role: 'zoom' }, ...(isMac ? [{ type: 'separator' }, { role: 'front' }] : [{ role: 'close' }])] },
  ]));
}

app.whenReady().then(() => {
  buildMenu();
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
