; StockAI_Dashboard Inno Setup Script
; See: https://jrsoftware.org/ishelp/

[Setup]
AppId={{D3F7A23C-7B6E-4F3D-9A7B-0A5C8E9B1234}}
AppName=StockAI Dashboard
AppVersion=0.0.0
AppPublisher=mejiruku
AppPublisherURL=https://github.com/mejiruku/StockAI_Dashboard
DefaultDirName={autopf}\StockAI_Dashboard
DefaultGroupName=StockAI Dashboard
AllowNoIcons=yes
; ビルドしたexeの場所を指定（PyInstallerの出力先 dist/StockAI_Dashboard.exe）
OutputDir=dist
OutputBaseFilename=StockAI_Dashboard_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=app_icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; メインの実行ファイル
Source: "dist\StockAI_Dashboard.exe"; DestDir: "{app}"; Flags: ignoreversion
; 必要なフォルダの作成
Source: "app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name: "{app}\temp"
Name: "{app}\reports"
Name: "{app}\models"

[Icons]
Name: "{group}\StockAI Dashboard"; Filename: "{app}\StockAI_Dashboard.exe"; IconFilename: "{app}\app_icon.ico"
Name: "{autodesktop}\StockAI Dashboard"; Filename: "{app}\StockAI_Dashboard.exe"; Tasks: desktopicon; IconFilename: "{app}\app_icon.ico"

[Run]
Filename: "{app}\StockAI_Dashboard.exe"; Description: "{cm:LaunchProgram,StockAI Dashboard}"; Flags: nowait postinstall skipfsgsilent
