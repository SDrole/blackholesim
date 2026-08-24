# blackholesim — Journal de session & feuille de route

> ⚠️ **PÉRIMÈTRE ÉLARGI le 2026-08-25** — la feuille de route active est
> désormais **`docs/ROADMAP_5_AXES.md`** (5 axes explicites + solveur GRMHD
> maison). Le présent document reste la référence pour l'historique de
> session et l'audit d'architecture polarisation/ingestion, mais son plan
> priorisé en §2 est **remplacé** par celui de ROADMAP_5_AXES.md.
> Changement notable : le point GRMHD n'est plus « ingestion seule » mais
> « solveur maison + ingestion comme format d'échange et référence de
> validation ».

Document de continuité : à relire en début de prochaine session pour reprendre le
projet sans perdre le contexte. Couvre (1) ce qui a été fait, (2) l'audit
d'architecture pour la suite (polarisation + GRMHD), (3) l'état courant.

---

## 1. Ce qui a été fait cette session

### Mise en place toolchain (machine locale, Windows, pas de GPU NVIDIA)
- Visual Studio 2026 Build Tools + CMake + Ninja installés via winget.
- Quirk découvert : `vcvars64.bat` ne fonctionne que si `vswhere.exe`
  (`...\Microsoft Visual Studio\Installer`) est déjà sur le PATH avant l'appel.
  Encodé dans `build_release.bat` à la racine.
- `/arch:AVX2` ajouté (gain modeste, ~7%, le rendu CPU est peu vectorisable
  automatiquement — beaucoup de branchements par rayon).

### Réécriture du disque CPU (Renderer.cpp)
- Ancien modèle : disque plan infiniment mince (un seul croisement du plan
  équatorial). Remplacé par un **tore volumétrique** avec transfert radiatif
  front-to-back (densité gaussienne verticale H(r)=0,05r à H/r réglable,
  intégration `dI = trans·S·(1-e^-dτ)`).
- **Bug trouvé et corrigé** : `diskSource()` appliquait un facteur `g⁴`
  explicite EN PLUS de celui déjà porté nativement par l'intégrale de Planck
  (`planckColor`, qui croît en ~T⁴ via Stefan-Boltzmann) → double comptage du
  beaming relativiste, saturation en blanc, texture noyée. Fix : retrait du
  g⁴ explicite côté CPU (la LUT Planck⊗CIE le porte déjà).
- Ajout d'une PSF/glow (convolution gaussienne à mélange pondéré, énergie
  conservée — pas un bloom "cinéma" arbitraire) : `Image::applyPointSpreadFunction`.

### Déploiement GPU (RunPod, A100 SXM, ~1,59 $/h)
- Repo poussé sur GitHub privé (`thaumaturgeafk/blackholesim`) — historique
  d'un ancien push (juin 2026) préservé sur la branche locale `old-remote-history`.
- Clé de déploiement SSH dédiée (lecture seule, un seul repo) enregistrée via
  `gh api repos/.../keys`.
- `CMakeLists.txt` : `CUDA_ARCHITECTURES` corrigé `89` (RTX 4090) → `80` (A100).
- **Piège RunPod découvert** : leur passerelle SSH (`ssh.runpod.io`) exige un
  PTY (`-tt`), ce qui pollue tout flux de données binaires avec des séquences
  d'échappement `ESC[?2004h/l` (bracketed paste mode) injectées en plein
  milieu du flux — y compris à l'intérieur d'une seule "ligne" `base64 -w0`.
  Solution de transfert de fichiers fiable : rediriger la sortie SSH vers un
  fichier local, isoler la ligne payload par longueur, puis
  `sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | tr -dc 'A-Za-z0-9+/='` avant
  `base64 -d`. Fonctionne jusqu'à au moins 33 Mo testés (BMP UHD bruts).
- `setup_a100.sh` à la racine : script one-shot (deps + build + rendu test).

### Trois fonctionnalités CUDA ajoutées (`src/cuda/Renderer.cu`)
1. **Turbulence MHD anisotrope** (`dev_mhdDensityFactor`, `dev_shearAspect`) :
   bruit étiré azimutalement par cisaillement képlérien réel
   S(r)=1,5Ω(r) — rapport d'aspect √(1+(Sτ)²) (résultat physique du tenseur
   de déformation d'un cisaillement simple, pas un facteur ad hoc). Cohérence
   transverse via label de flux poloïdal ψ(r)∝r^0,75.
2. **Transfert radiatif multi-bande** (`dev_opacitySpectrum`) : transmittance
   RGB vectorielle (opacité chromatique par canal, indice spectral réglable),
   au lieu d'un scalaire gris partagé entre les 3 canaux.
3. **Anneaux de photons d'ordre élevé n≥2,3** : corridor de pas raffiné dans
   toute la sphère de photons (pas seulement le plan du disque), budget de
   pas dédié GPU (`gpuMaxSteps`), et **sur-échantillonnage adaptatif** dans
   le kernel (détection de bord fond/disque/horizon → rayons supplémentaires
   ciblés). Bug de terminaison prématurée corrigé (rayon quasi-critique qui
   épuise son budget de pas était classé "fond noir" à tort).

### Bug critique trouvé sur le premier vrai test A100
Le pipeline couleur CUDA (`dev_planckB`/`dev_planckRGB`) n'avait **jamais été
calibré en luminance** — contrairement à la LUT CPU (normalisée Y=1 au pic),
il retournait des valeurs ~40× trop faibles à température de disque typique
(~7000 K), sous-exposant tout le rendu GPU. Fix : `p.planckNorm`, calibré
côté hôte pour que `B(546nm, diskTempScaleK) = 1`, même convention que le
CPU. Trouvé et corrigé en conditions réelles (premier rendu GPU trop sombre),
pas en amont — leçon : le portage CUDA n'avait jamais été testé faute de GPU
disponible avant cette session.

### Réglages qualité poussés au maximum (Config.hpp)
`gpuMaxSteps` 600k→1,5M, `photonRingStep` 0,01→0,005, `ringSupersample` 4→8,
`diskShearWindup` 2,5→4,0, `diskOpacitySpectralIndex` 1,5→2,0.

### Livrables produits
- 3 images UHD (3840×2160, AA=4) : spin 0 (Schwarzschild), 0,9, 0,998
  (quasi-extrémal) — BMP bruts + JPEG aperçu dans `output/uhd/`.
  Note : le rendu spin=0,998 est visiblement plus sombre que les deux
  autres — probablement réel (ISCO à 1,237M seulement de l'horizon à
  1,063M, redshift gravitationnel extrême sur le bord interne), non
  ré-examiné en détail, à creuser si besoin plus tard.
- Vidéo 900 images, 1280×720, AA=2, spin=0,9, assemblée en 60 fps/15s
  (`blackhole_60fps_15s.mp4` sur le pod). Rendu terminé en 11469,6 s
  (3h11), Carter conservée (dérive max 1,44e-7).

---

## 2. Audit d'architecture — prochaine étape (polarisation + GRMHD)

Discussion complète menée et tranchée cette session. Résumé actionnable :

### Décisions prises
- **Rejet formel** de l'option "écrire notre propre solveur GRMHD" — la
  communauté a déjà produit et validé ce travail (iharm3D, BHAC), un solveur
  maison n'apporterait que du risque numérique pour zéro valeur ajoutée.
- **Format d'ingestion GRMHD retenu : iharm HDF5** — standard de facto de
  l'écosystème EHT, vérifié : `ipole` le lit nativement
  (`AFD-Illinois/ipole`, `model/iharm/model.c`), et un convertisseur
  BHAC→iharm existe déjà (`AFD-Illinois/EHT-babel`). Pas de format à
  inventer.
- **Ordre de travail contre-intuitif mais nécessaire** : le module
  d'ingestion GRMHD (Track B) doit précéder la validation sérieuse de la
  polarisation (Track A), car comparer notre champ B "jouet" à une sortie
  `ipole` basée sur un vrai GRMHD ne validerait rien — il faut la même
  source physique des deux côtés pour qu'une comparaison ait un sens.

### Architecture retenue (5 modules, découpage validé)
```
1. Géométrie/géodésiques      → EXISTE DÉJÀ (Geodesic.cpp, Metric.cpp, Integrator.hpp)
2. Plasma local                 → NOUVELLE INTERFACE (le point clé, cf. ci-dessous)
3. Coefficients synchrotron    → NOUVEAU, pure fonction de l'état plasma
4. Solveur transfert polarisé    → NOUVEAU, risque technique principal
5. Projection observateur        → EXTENSION du code caméra existant
```

Interface centrale (jonction entre Track A et Track B, à concevoir une
seule fois) :
```cpp
struct PlasmaState {
    double rho;         // densité (unités géométrisées code)
    double Theta_e;      // température électronique adimensionnée kT_e/(m_e c²)
    double u4[4];         // quadrivitesse fluide (contravariant)
    double B4[4];          // quadrivecteur champ magnétique (contravariant)
    bool   valid;
};
// Deux backends interchangeables (switch sur enum en CUDA, pas de vtable device) :
//   sampleProcedural(...)     — modèle actuel (fBm cisaillé)
//   sampleGRMHDSnapshot(...)  — interpolation dans un snapshot iharm HDF5
```

### Variables minimales pour un snapshot GRMHD (confirmé)
`RHO`, `UU` (énergie interne), `U1/U2/U3` (vitesse *primitive* HARM — pas
directement les composantes de la quadrivitesse, reconstruction standard
nécessaire), `B1/B2/B3` (champ magnétique primitif, même remarque) + grille
FMKS (Kerr-Schild modifié, **différent** de notre Boyer-Lindquist actuel —
transformation analytique à implémenter et tester explicitement).

### Pièges techniques identifiés (à ne pas rater)
- **RK4 naïf sur le transfert polarisé = dangereux**, pas juste "moins
  précis" : ces équations sont raides (forte rotation de Faraday possible
  sur un seul pas). La communauté utilise un schéma semi-analytique par
  segment (Mościbrodzka & Gammie 2018, repris dans `ipole`) précisément
  pour cette raison.
- Reconstruction primitives HARM → quadrivitesse/champ B réels (erreur
  classique du domaine si prise pour argent comptant).
- Convention d'angle de polarisation (EVPA, IAU) — source de bugs
  documentés historiquement dans plusieurs codes publiés.
- Unités du champ B (Gaussien vs Heaviside-Lorentz, facteur √4π).

### Cible de validation numérique (vérifiée, pas inventée)
Papier de comparaison inter-codes EHT (arXiv:2303.12004) : NMSE ≈ 0,01-0,02
sur Stokes I/Q/U entre codes indépendants validés, jusqu'à ≈0,12 sur V (V
intrinsèquement plus bruité). **Pas d'égalité bit-à-bit** — c'est le seuil
que la communauté elle-même utilise pour dire "deux codes indépendants sont
d'accord".

### Versions proposées (MVP → publication)
1. **Minimale** : vraies formules synchrotron (Pandya) en Stokes I seul, sur
   backend procédural (champ B jouet) — teste le module 3 isolément.
2. **Robuste** : polarisation complète I,Q,U,V avec solveur semi-analytique,
   toujours sur backend procédural — validée par cas analytiques + tendance
   qualitative vs `ipole`.
3. **Standard publication** : polarisation complète + backend GRMHD réel,
   comparaison directe à `ipole` sur le **même** snapshot d'entrée, cible
   NMSE ~0,01-0,05 par paramètre de Stokes.

### Plan de travail priorisé
1. ✅ **FAIT** (2026-08-25) — Interface `PlasmaState` + backend procédural,
   **CPU ET CUDA**.
   - `src/Plasma.hpp` (nouveau) : struct POD partagée host/device.
   - CPU : `Renderer::sampleProceduralPlasma()`, `traceRay` route la densité
     à travers l'interface au lieu d'appeler `diskDensity()` directement.
     Vérifié bit-identique (dérive de Carter 2,82929e-08 sur la config de
     référence 480x270/AA=1/spin=0.9). Commit `875ccf7`.
   - CUDA : `dev_sampleProceduralPlasma()`, miroir exact du CPU, `dev_traceRay`
     branché dessus. Commit `d63e6ee`. **NON COMPILÉ** (pas de nvcc en local)
     — à builder sur GPU avant de faire confiance, notamment l'espace
     d'exécution du constructeur par défaut implicite de `PlasmaState`.
   - Les deux backends peuplent aussi u^mu (orbite képlérienne circulaire
     exacte, formules déjà existantes) et un champ B4 JOUET (toroïdal, non
     utilisé par le pipeline avant les étapes 3-4).

   ⚠️ **Leçon** : le premier passage n'avait fait que le CPU, laissant le
   chemin GPU (celui des vrais rendus) hors de l'interface. Garder CPU et
   CUDA structurellement parallèles n'est pas cosmétique — c'est exactement
   la divergence CPU/CUDA qui a produit les DEUX bugs physiques de cette
   session. À vérifier systématiquement pour chaque étape suivante.
2. Module synchrotron (Pandya) en Stokes I seul, sur backend procédural.
3. Module d'ingestion iharm HDF5 + backend GRMHD (source de snapshot à
   déterminer à ce moment : bibliothèque théorique EHT — accès en masse non
   confirmé, à revérifier — ou faire tourner `iharm3D` nous-mêmes sur GPU
   loué en filet de sécurité).
4. Polarisation complète (solveur semi-analytique) sur backend GRMHD.
5. Validation comparative à `ipole` sur snapshot partagé.

### Sources citées (vérifiées par recherche cette session)
- ipole : https://github.com/AFD-Illinois/ipole (BSD-3-Clause)
- Comparaison codes polarisés EHT : https://arxiv.org/html/2303.12004
- Vérification transfert non-polarisé (Gold et al. 2020) :
  https://eventhorizontelescope.org/publications/verification-radiative-transfer-schemes-eht
- Pandya, Zhang, Chandra & Gammie 2016 : https://iopscience.iop.org/article/10.3847/0004-637X/822/1/34
- iharm3D : https://github.com/AFD-Illinois/iharm3d
- EHT-babel (convertisseur format) : https://github.com/AFD-Illinois/EHT-babel

---

## 3. État courant / prochaine session

- Vidéo 900 images terminée, assemblage ffmpeg 60fps en cours sur le pod
  au moment de la rédaction de ce document.
- **Le pod A100 tourne toujours et coûte 1,59 $/h — à arrêter dès que la
  vidéo est récupérée**, sauf si la prochaine étape (§2) démarre dans la
  foulée.
- Prochaine session : reprendre au point 1 du plan de travail priorisé
  ci-dessus (interface `PlasmaState`), sauf changement de priorité décidé
  entretemps.
