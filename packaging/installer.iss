; Inno Setup Script for PDF to MuseScore
; Supports Windows 10 (Home and upwards) and Windows 11

#define MyAppName "PDF to MuseScore"
#define MyAppVersion "0.6.0"
#define MyAppPublisher "Tomer Mayost"
#define MyAppCredit "By Tomer Mayost"
#define MyAppExeName "pdf2musescore.exe"

[Setup]
AppId={{8D4F61C0-5A2E-4E91-9A23-7B8C1F42E101}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppCopyright={#MyAppCredit}
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} ({#MyAppCredit})
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\tmp\dist
OutputBaseFilename=pdf2musescore-{#MyAppVersion}-setup-x64
Compression=zip
WizardStyle=modern
SetupIconFile=..\src\pdf2musescore\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\tmp\dist\pdf2musescore\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyAppCredit}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyAppCredit}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--download-models"; StatusMsg: "Downloading AI music-recognition models (~180 MB, one time)..."; Flags: waituntilterminated; Check: not WizardSilent
Filename: "{app}\{#MyAppExeName}"; Parameters: "--download-models --no-ui"; StatusMsg: "Downloading AI music-recognition models..."; Flags: waituntilterminated runhidden; Check: WizardSilent
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\pdf2musescore\work"
Type: filesandordirs; Name: "{commonappdata}\pdf2musescore\work"

[Code]
const
  StorePackagesSubkey = 'Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages';
  StorePackagePrefix = 'TomerMayost.PDF2MuseScore_';
  InnoUninstallSubkey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{8D4F61C0-5A2E-4E91-9A23-7B8C1F42E101}_is1';

function ParseVersionPart(var Rest: String): Integer;
var
  DotPos: Integer;
  Part: String;
begin
  Rest := Trim(Rest);
  if Rest = '' then begin
    Result := 0;
    Exit;
  end;
  DotPos := Pos('.', Rest);
  if DotPos > 0 then begin
    Part := Copy(Rest, 1, DotPos - 1);
    Delete(Rest, 1, DotPos);
  end else begin
    Part := Rest;
    Rest := '';
  end;
  Result := StrToIntDef(Trim(Part), 0);
end;

function CompareVersionStrings(const A, B: String): Integer;
var
  RemA, RemB: String;
  PartA, PartB, I: Integer;
begin
  RemA := Trim(A);
  RemB := Trim(B);
  if (Length(RemA) > 0) and ((RemA[1] = 'v') or (RemA[1] = 'V')) then
    Delete(RemA, 1, 1);
  if (Length(RemB) > 0) and ((RemB[1] = 'v') or (RemB[1] = 'V')) then
    Delete(RemB, 1, 1);
  Result := 0;
  for I := 1 to 4 do begin
    PartA := ParseVersionPart(RemA);
    PartB := ParseVersionPart(RemB);
    if PartA > PartB then begin
      Result := 1;
      Exit;
    end;
    if PartA < PartB then begin
      Result := -1;
      Exit;
    end;
  end;
end;

function FormatShortVersion(const Ver: String): String;
var
  I, Dots: Integer;
begin
  Result := Trim(Ver);
  Dots := 0;
  for I := 1 to Length(Result) do begin
    if Result[I] = '.' then
      Dots := Dots + 1;
  end;
  if (Dots = 3) and (Length(Result) > 2) and (Copy(Result, Length(Result) - 1, 2) = '.0') then
    Delete(Result, Length(Result) - 1, 2);
end;

function GetStandaloneInstalledVersion(): String;
var
  Ver: String;
begin
  Result := '';
  if RegQueryStringValue(HKEY_CURRENT_USER, InnoUninstallSubkey, 'DisplayVersion', Ver) then begin
    Result := Trim(Ver);
    Exit;
  end;
  if RegQueryStringValue(HKEY_LOCAL_MACHINE, InnoUninstallSubkey, 'DisplayVersion', Ver) then
    Result := Trim(Ver);
end;

function GetStoreInstalledVersion(): String;
var
  Subkeys: TArrayOfString;
  I, UndPos: Integer;
  KeyName, Tail, Ver: String;
  PrefixLower: String;
begin
  Result := '';
  PrefixLower := Lowercase(StorePackagePrefix);
  if not RegGetSubkeyNames(HKEY_CURRENT_USER, StorePackagesSubkey, Subkeys) then
    Exit;
  for I := 0 to GetArrayLength(Subkeys) - 1 do begin
    KeyName := Subkeys[I];
    if Pos(PrefixLower, Lowercase(KeyName)) = 1 then begin
      Tail := Copy(KeyName, Length(StorePackagePrefix) + 1, Length(KeyName));
      UndPos := Pos('_', Tail);
      if UndPos > 1 then begin
        Ver := Copy(Tail, 1, UndPos - 1);
        if (Result = '') or (CompareVersionStrings(Ver, Result) > 0) then
          Result := Ver;
      end;
    end;
  end;
end;

function InitializeSetup(): Boolean;
var
  ExistingStandalone, ExistingStore: String;
  StoreDisplay, Msg: String;
  CmpStore: Integer;
begin
  Result := True;

  ExistingStandalone := GetStandaloneInstalledVersion();
  if (ExistingStandalone <> '') and (CompareVersionStrings(ExistingStandalone, '{#MyAppVersion}') > 0) then begin
    Msg := 'A newer standalone version (' + FormatShortVersion(ExistingStandalone) + ') of {#MyAppName} is already installed.' + #13#10 + #13#10 +
           'Downgrading to version {#MyAppVersion} is not supported. Uninstall the newer version first if you intentionally need an older release.';
    Log('Blocking standalone downgrade: installed=' + ExistingStandalone + ', installer={#MyAppVersion}');
    if not WizardSilent then
      MsgBox(Msg, mbError, MB_OK);
    Result := False;
    Exit;
  end;

  ExistingStore := GetStoreInstalledVersion();
  if ExistingStore <> '' then begin
    StoreDisplay := FormatShortVersion(ExistingStore);
    CmpStore := CompareVersionStrings(ExistingStore, '{#MyAppVersion}');
    if CmpStore > 0 then begin
      Msg := 'A newer Microsoft Store version (' + StoreDisplay + ') of {#MyAppName} is already installed on this PC.' + #13#10 + #13#10 +
             'Downgrading by installing standalone version {#MyAppVersion} is not supported. Launch PDF2MuseScore from the Start Menu to use the Store edition.';
      Log('Blocking downgrade over newer Microsoft Store edition: store=' + ExistingStore + ', installer={#MyAppVersion}');
      if not WizardSilent then
        MsgBox(Msg, mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if CmpStore = 0 then begin
      Msg := 'The Microsoft Store edition of {#MyAppName} (version ' + StoreDisplay + ') is already installed on this PC.' + #13#10 + #13#10 +
             'Installing the standalone .exe alongside it will create duplicate Start Menu entries, and only the Store edition receives automatic Store updates.' + #13#10 + #13#10 +
             'Do you still want to continue with this standalone installation?';
    end else begin
      Msg := 'The Microsoft Store edition of {#MyAppName} (version ' + StoreDisplay + ') is already installed on this PC.' + #13#10 + #13#10 +
             'Installing standalone version {#MyAppVersion} alongside it will create a second copy. You can check the Microsoft Store app for updates instead, or remove one copy in Installed Apps later.' + #13#10 + #13#10 +
             'Do you still want to continue with this standalone installation?';
    end;

    Log('Detected existing Microsoft Store edition (' + ExistingStore + ') during {#MyAppVersion} setup');
    if not WizardSilent then begin
      if MsgBox(Msg, mbConfirmation, MB_YESNO or MB_DEFBUTTON2) <> IDYES then begin
        Result := False;
        Exit;
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  StoreVer, LocalModels, CommonModels: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    StoreVer := GetStoreInstalledVersion();
    if StoreVer <> '' then begin
      Log('Preserving shared AI models because Microsoft Store edition (' + StoreVer + ') is installed.');
      Exit;
    end;
    LocalModels := ExpandConstant('{localappdata}\pdf2musescore\models');
    CommonModels := ExpandConstant('{commonappdata}\pdf2musescore\models');
    if DirExists(LocalModels) then
      DelTree(LocalModels, True, True, True);
    if DirExists(CommonModels) then
      DelTree(CommonModels, True, True, True);
  end;
end;
