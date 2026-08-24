# blackholesim — Roadmap complète (5 axes de réalisme scientifique)

Document de référence de la feuille de route. Décidé le 2026-08-25.
Complète `SESSION_LOG_AND_ROADMAP.md` (qui garde l'historique de session et
l'audit d'architecture polarisation/ingestion).

**Changement de périmètre majeur (2026-08-25)** : le point 5 passe de
« ingestion de snapshots publics » à « **développer notre propre solveur
GRMHD auto-cohérent** », séparé du ray tracer, en s'appuyant sur les méthodes
et références open source existantes. L'ingestion iharm HDF5 reste développée
— elle devient le format d'échange entre notre solveur et blackholesim, ET le
moyen de valider notre solveur contre des codes établis.

---

## Les 5 axes

| # | Axe | Dépendances | Effort |
|---|---|---|---|
| 1 | Variabilité temporelle + diffusion interstellaire | aucune | faible |
| 2 | Température électronique (R-high/R-low) | PlasmaState ✅ | faible |
| 3 | Synchrotron réel à 230 GHz | axe 2 | moyen |
| 4 | Polarisation complète I,Q,U,V + Faraday | axe 3 + vrai champ B | élevé |
| 5 | Solveur GRMHD maison auto-cohérent | — (piste longue parallèle) | très élevé |

## Briques transversales

| Brique | Rôle | État |
|---|---|---|
| `PlasmaState` | interface unique plasma → rendu (CPU + CUDA) | ✅ fait |
| Lecteur iharm HDF5 | snapshots publics + sortie de notre solveur | à faire |
| Écrivain iharm HDF5 | sortie de notre solveur GRMHD | à faire (axe 5) |
| Validation vs `ipole` | valide le transfert radiatif polarisé | à faire (axe 4) |
| Validation GRMHD | valide notre solveur (cas-tests + Porth 2019) | à faire (axe 5) |

---

## Ordre d'exécution recommandé

Principe : l'axe 5 est la piste longue (mois) ; les axes 1-4 sont des gains
en semaines. On avance les deux **en parallèle** plutôt qu'en série, et on
place tôt les briques qui débloquent la validation des autres.

**Phase A — gains rendu rapides (indépendants du GRMHD)**
1. Axe 1 : variabilité + diffusion interstellaire (le moins cher, autonome).
2. Axe 2 : `Theta_e` via prescription R-high/R-low (peuple le champ déjà
   réservé dans `PlasmaState`).
3. Axe 3 : synchrotron 230 GHz, Stokes I seul, coefficients Pandya.

**Phase B — vrai champ magnétique (débloque la polarisation)**
4. Lecteur iharm HDF5 + backend `sampleGRMHDSnapshot` → charger des
   snapshots **publics** : premier vrai champ B, et référence pour valider
   plus tard notre propre solveur.
5. Axe 4 : polarisation complète sur ces snapshots publics, validée contre
   `ipole` sur la **même** source physique.

**Phase C — solveur GRMHD maison (démarrable en parallèle dès maintenant)**
6. Axe 5 : voir la stratégie détaillée ci-dessous. Nos snapshots remplacent
   progressivement les snapshots publics une fois validés.

Rationale : la Phase B donne un champ B réel bien avant que notre solveur
soit prêt — donc l'axe 4 n'attend pas l'axe 5. Et disposer de snapshots
publics **avant** de finir notre solveur est ce qui rend sa validation
possible (comparaison directe sur le même problème).

---

## AXE 5 — Stratégie du solveur GRMHD

### Contrainte d'architecture (fixée)

```
[ grmhd/ solveur séparé ]  →  snapshots HDF5 (format iharm)  →  [ blackholesim ]
   exécutable autonome          format d'échange standard        ray tracer + transfert
```

Aucun couplage direct : le solveur ne connaît pas le ray tracer, le ray
tracer ne connaît que le format de fichier. Cela permet de valider chacun
indépendamment, et de continuer à consommer des snapshots publics.

### Stratégie de licence (vérifiée 2026-08-25)

| Code | Licence | Utilisation possible |
|---|---|---|
| Athena++ | BSD-3 | adaptation de code OK (avec attribution) |
| KHARMA | BSD-3 | adaptation de code OK (avec attribution) |
| iharm3D | BSD-3 (org AFD-Illinois ; **à reconfirmer fichier LICENSE**) | probablement OK |
| BHAC | **GPL v3** | ⚠️ copier son code rendrait blackholesim GPL |

**Principe clé** : les *équations et méthodes numériques* publiées dans les
articles ne sont pas soumises au droit d'auteur — s'en inspirer n'impose
aucune obligation. Seule la *copie de code source* est concernée. Donc :
lire BHAC pour comprendre = libre ; copier ses lignes = contamination GPL.
Privilégier Athena++/KHARMA/iharm3D si on veut adapter du code réel.

### Briques à construire, dans l'ordre

> **Découpage révisé le 2026-08-25.** La version initiale mélangeait hydro et
> MHD : elle testait l'inversion « en forte magnétisation » (S2) alors que le
> champ B n'arrivait qu'en S5, et proposait les tubes à choc de Komissarov
> (qui sont **SRMHD**) pour un cœur annoncé hydro (S3). Le découpage ci-dessous
> sépare proprement hydro → GR hydro → MHD → GRMHD.
>
> Précision importante : l'inversion U→P **hydro** (S2) et l'inversion
> **magnétisée** (S5) ne sont PAS la même brique à un terme près — ce sont deux
> algorithmes différents, le second nettement plus dur (c'est celui de Noble et
> al. 2006, là où les codes cassent réellement).

**S0 ✅ FAIT + S1 ✅ FAIT & CONSOLIDÉ (2026-08-25)** — `grmhd/` : `Grid.hpp`,
`Coordinates.hpp` (MKS), `KerrSchild.hpp/.cpp`, `LinAlg.hpp`, `tests_s1.cpp`,
`tests_s1_consolidation.cpp`. Cibles CMake `grmhd_tests` et
`grmhd_s1_consolidation`. Reste ouvert : backend HDF5 (contrainte ferme
ci-dessus), écriture d'état fluide (rien à écrire tant qu'il n'y a pas de
fluide).

**Résultats de consolidation S1 — la géométrie est verrouillée :**

*Bloc A — schéma de différentiation des Christoffels (validé contre un oracle
indépendant : différentiation par PAS COMPLEXE `Im[f(x+ih)]/h`, exacte à la
précision machine car sans soustraction donc sans annulation catastrophique).*
- Différences centrées vs pas complexe : `∂_r g` 7,6e-11, `∂_θ g` 1,6e-10.
- Balayage du pas `h` sur 10 décades : courbe en V classique, optimum
  empirique à `h≈1e-4` (8,5e-12). Le `h=1e-6` de production est ~10× au-dessus
  de l'optimum, donc **dans le plateau** — validé, pas sur un flanc.
- **Proximité de l'axe polaire — contrainte de maillage découverte** :
  `∂_θ g` se dégrade régulièrement près de l'axe (1,4e-9 à 1e-2 rad ;
  1,7e-8 à 1e-3 ; 1,1e-6 à 1e-5) car les points `θ±h` finissent par franchir
  l'axe, où `|sinθ|` réfléchit la fonction au lieu de la prolonger.
  **Borne sûre MESURÉE : ≥ 1e-2 rad** (ma borne initiale de 1e-3 était
  optimiste d'une décade — corrigée sur la donnée, pas en élargissant la
  tolérance). Vérifié que le MKS la respecte largement : 1re cellule à
  1,04e-2 rad même à n2=256. ⚠️ **À revérifier si on augmente la résolution
  polaire au-delà de n2≈256, ou si on change `hslope`.**

*Bloc B — validation croisée contre le ray tracer Boyer-Lindquist* (oracle
INDÉPENDANT : formalisme hamiltonien canonique, code de production déjà
validé par la conservation de Carter). `src/Metric.cpp` et `src/Geodesic.cpp`
sont liés directement dans la cible de test.
- Accord KS↔BL sur `(r, θ)` à λ=20 : **1,5e-12** sur 5 géodésiques
  (spins 0 / 0,5 / 0,9 / 0,998 ; équatoriales et inclinées ; dont une proche
  horizon). `r` et `θ` étant des coordonnées PARTAGÉES entre BL et KS, la
  comparaison est directe — aucune transformation dans le test, donc aucun
  risque d'erreur introduite par le test lui-même.
- Invariants KS (E, Lz, nullité) : 4,6e-11. Carter côté BL : 3,4e-15.
- **Convergence : ordre observé +4,04** dans le régime de troncature
  (dl 0,8 → 0,4), puis dégradation vers le plancher ~1e-12 imposé par la
  différentiation numérique des Christoffels. Conforme à RK4.

> ⚠️ **Second piège méthodologique (même famille que la tautologie T3).** Le
> test de convergence a d'abord été dimensionné à partir de dl=0,02 : l'écart
> y valait déjà ~1e-12, c'est-à-dire le plancher numérique. Les « ordres »
> mesurés oscillaient (+2,7 / −2,2 / +2,3) — **on mesurait du bruit d'arrondi
> en croyant mesurer une convergence**. Corrigé en partant de pas nettement
> plus grossiers et en n'évaluant l'ordre que sur les paires dont les deux
> erreurs dépassent franchement le plancher.
>
> Règle à appliquer systématiquement : **un test de convergence n'est valide
> que dans le régime où l'erreur de troncature domine** — il faut le vérifier
> explicitement, pas le supposer.

**S0 — Infrastructure & sortie**
Répertoire `grmhd/`, cible CMake séparée, structures de grille, interface
d'écriture d'état. *Fait en premier volontairement* : pouvoir dumper et
inspecter est l'outil de debug principal de tout ce qui suit.
*Note dépendance* : HDF5 n'est pas installé en local (absent de winget ;
conda est la voie). À S0-S1 il n'y a aucun état fluide à écrire — backend
binaire/CSV sans dépendance d'abord.

> 🔒 **CONTRAINTE FERME (décidée 2026-08-25)** — le writer **iharm HDF5 doit
> être opérationnel au plus tard PENDANT S3**, avant qu'on accumule des états
> fluides significatifs, et **obligatoirement avant de commencer S4**.
> Raison : ne pas repousser cette dette technique au moment où le solveur
> sera déjà complexe — changer de format de sortie sur un code mûr coûte
> beaucoup plus cher, et sans écriture standard on ne peut ni inspecter
> sérieusement, ni comparer aux codes de référence, ni alimenter
> blackholesim. Cette contrainte prime sur toute tentation de « on verra à
> S5 ».

**S1 — Métrique & coordonnées (Kerr-Schild)**
⚠️ Différence importante avec l'existant : le ray tracer utilise
Boyer-Lindquist (singulier à l'horizon). Un solveur GRMHD a besoin de
**Kerr-Schild** (pénétrant l'horizon) + transformation MKS (maillage log en
r, concentration en θ vers le plan équatorial).
Livrables : `g_μν`, `g^μν`, `√-g`, symboles de Christoffel.
*Tests* : `g·g⁻¹ = I` ; `√-g` vs formule analytique `Σ sinθ` (validation
indépendante de la métrique covariante) ; symétrie de Γ ; stationnarité +
axisymétrie ; régularité à l'horizon (composantes finies en `r₊`,
contrairement à BL) ; **conservation des quantités de Killing `p_t`, `p_φ` et
du caractère nul le long de géodésiques intégrées** (le vrai test de la
connexion).

> ⚠️ **Piège méthodologique rencontré et corrigé (2026-08-25).** La version
> initiale annonçait la compatibilité métrique `∇_λ g_μν = 0` comme « le vrai
> test de la connexion ». C'est **faux** : en substituant la définition de Γ,
> on obtient identiquement `∂_λ g_μν - ∂_λ g_μν = 0` — la relation est une
> **tautologie algébrique**, vraie même avec des dérivées de métrique
> complètement fausses. Elle ne détecte qu'une faute de transcription dans la
> formule de Γ elle-même.
>
> Le symptôme qui a trahi le problème : le test rendait 8e-16 (précision
> machine) là où la différentiation numérique impose un plancher ~1e-9.
> **Un test meilleur que sa limite théorique ne teste pas ce qu'on croit.**
> Conservé comme T3 (il garde une valeur : détection de faute de frappe dans
> Γ), mais le test réel est désormais T8 (géodésiques), qui rend 1e-13…1e-11
> — dominé par de vraies erreurs numériques, signature d'un test authentique.
>
> Règle générale à appliquer aux étapes suivantes : **se méfier d'un test qui
> passe trop bien**, et vérifier algébriquement qu'il n'est pas satisfait par
> construction.

**S2 — Primitives ↔ conservées, HYDRO relativiste (sans champ B)**
`P→U` analytique ; `U→P` par recherche de racine 1D — docile comparé au cas
magnétisé.
*Test* : aller-retour `P→U→P` sur un large domaine, **facteurs de Lorentz
élevés** (pas de magnétisation ici : elle n'existe pas encore). Test
unitaire pur, aucune simulation.

**S3 — Solveur SRHD en espace plat + tubes à choc**
Reconstruction PLM (minmod/MC), solveur de Riemann HLL, intégrateur RK2/RK3
SSP. Hydro pur, métrique plate.
*Tests* : problèmes de Riemann relativistes de **Martí & Müller 1994**
(solution analytique exacte — PAS Komissarov, qui est magnétisé).
⚠️ *Point d'architecture* : écrire le solveur **agnostique de la métrique**
dès ici (l'espace plat = un choix de métrique parmi d'autres). Sinon les
tests SRMHD de S5 exigeront un second chemin de code.
⚠️ *Planchers* : des planchers de positivité minimaux (ρ, p > 0) apparaîtront
naturellement dès ce stade dans les raréfactions fortes. C'est normal et
inévitable — S6 reste la brique *systématique*, pas le premier plancher.

**S4 — GRHD en Kerr-Schild (termes sources GR, toujours sans B)**
*Tests décisifs* :
- **Bondi/Michel** : accrétion sphérique stationnaire, **solution analytique
  exacte**. Doit rester stationnaire. Test le plus révélateur d'une erreur
  dans les termes sources ou la métrique.
- **Tore de Fishbone-Moncrief sans champ B** : équilibre hydro exact, doit
  rester statique. Toute dérive = bug.

**S5 — Ajout complet du magnétisme (SRMHD → GRMHD)**
Quatre sous-briques distinctes :
1. Variables GRMHD (ajout de `B^i` aux primitives, du terme de Maxwell au
   tenseur énergie-impulsion).
2. **Inversion magnétisée** `U→P` — la brique difficile (Noble et al. 2006).
   *Test* : aller-retour à **forte magnétisation** ET facteur de Lorentz
   élevé (c'est ici que ce test a sa place, pas en S2).
3. Équation d'induction + **transport contraint** FLUX-CT (Tóth 2000).
   *Test* : `∇·B` reste au niveau de la précision machine sur des milliers
   de pas. Invariant non négociable, monitoré en permanence.
4. *Tests d'ensemble* : suite de tubes à choc **SRMHD de Komissarov** (en
   espace plat — gratuit si S3 est bien métrique-agnostique).

**S6 — Planchers, plafonds, conditions aux limites (traitement systématique)**
Planchers de densité/énergie interne, plafond de magnétisation σ et de
facteur de Lorentz ; sorties libres en r, axe polaire, périodicité en φ.
Ce n'est pas de la physique mais c'est **indispensable** : sans ça, NaN
garantis dans la région du jet/funnel. Tous les codes du domaine en ont.
*Diagnostic* : compteurs d'activation des planchers — un plancher qui se
déclenche massivement signale un problème physique en amont.

**S7 — Tore magnétisé + MRI, puis validation communautaire**
Tore de Fishbone-Moncrief + boucle poloïdale faible → turbulence MRI.
*Tests* :
- **Taux de croissance linéaire de la MRI vs prédiction analytique**
  (quantitatif, pas visuel).
- Puis métriques de **Porth et al. 2019** (comparaison de 9 codes :
  Athena++, BHAC, iharm3D, KORAL, H-AMR…) : taux d'accrétion, flux
  magnétique à l'horizon, séries temporelles. Standard de validation le plus
  élevé accessible.

### ⚠️ 2D d'abord, mais 3D obligatoire à terme

Développer et valider en **2D axisymétrique** (bien moins cher, la plupart
des tests fonctionnent). Mais limite physique fondamentale : le **théorème
anti-dynamo de Cowling** interdit d'entretenir une turbulence MRI en
axisymétrie — elle décroît nécessairement. Le 2D sert au développement et à
la validation ; une turbulence entretenue exige le 3D. À ne pas confondre
avec un bug le jour où la turbulence 2D s'éteindra : c'est attendu.

### Quand des snapshots deviennent exploitables par blackholesim ?

- **Après S4** : snapshots hydro GR (densité, vitesse) **sans champ B** —
  déjà utilisables par le chemin thermique/Planck actuel du rendu.
- **Après S5–S7** : snapshots magnétisés → synchrotron et polarisation
  réels.

Écrire le format HDF5 dès S0 permet aussi de brancher blackholesim très tôt
sur des données intermédiaires, ne serait-ce que pour vérifier visuellement
la cohérence.

---

## Protocole anti-« ça tourne mais la physique est fausse »

Le vrai risque n'est pas le crash — c'est un code qui produit une image
plausible et fausse. Règles :

1. **Aucune brique n'avance sans passer son test.** Pas de « on verra plus
   tard », c'est ainsi qu'on accumule des bugs qui se compensent.
2. **Solutions analytiques partout où elles existent** : Bondi/Michel, taux
   de croissance MRI, tubes à choc. Ce sont des vérités indépendantes du
   code, pas des auto-confirmations.
3. **Invariants surveillés en permanence à l'exécution** — exactement comme
   la vérification de la constante de Carter déjà présente dans le ray
   tracer (bon précédent maison) : conservation masse/énergie/moment
   cinétique, `∇·B`, compteurs d'activation des planchers (un plancher qui
   se déclenche massivement signale un problème physique en amont).
4. **Tests de convergence** : en raffinant la grille, l'erreur doit décroître
   à l'ordre attendu. Un bug passe souvent inaperçu à une seule résolution.
5. **« Ce qui est stationnaire doit rester stationnaire »** — le détecteur de
   bug le plus sensible du lot.
6. **Comparaison à des résultats publiés** sur le même problème (Porth 2019).
7. **Ne jamais accepter la plausibilité visuelle comme validation.** « Ça
   ressemble à un trou noir » ne prouve rien. Cette session a déjà produit
   deux bugs physiques réels (double comptage du beaming g⁴, calibration de
   luminance CUDA) qui donnaient des images parfaitement plausibles.

---

## Ampleur réelle du chantier (honnêteté)

iharm3D fait ~10 000 lignes de C ; un solveur GRMHD 2D minimal mais correct
représente plusieurs milliers de lignes **plus** l'essentiel du temps en
validation. Compter en **mois** de travail à temps partiel, pas en semaines.
La difficulté n'est pas d'écrire les équations — c'est d'obtenir un schéma
stable (inversion U→P, planchers, CT) et de prouver qu'il est juste.

C'est un objectif assumé (challenge + adaptabilité), pas le chemin le plus
court vers de belles images : les axes 1-4 avec snapshots publics donnent un
bien meilleur rapport résultat/effort à court terme. Les deux pistes
avancent en parallèle, précisément pour cette raison.

---

## Références vérifiées

- Porth et al. 2019, *EHT GRMHD Code Comparison Project* — https://arxiv.org/abs/1904.04923
- Noble et al. 2006 (inversion U→P) — référence standard du domaine
- Tóth 2000 (constrained transport / FLUX-CT)
- Komissarov (suite de tubes à choc SRMHD)
- Athena++ — https://arxiv.org/pdf/2005.06651 (BSD-3)
- KHARMA — https://github.com/AFD-Illinois/kharma (BSD-3)
- iharm3D — https://github.com/AFD-Illinois/iharm3d
- BHAC — https://bhac.science (GPL v3)
- Pandya et al. 2016 (synchrotron) — https://arxiv.org/abs/1602.08749
- ipole — https://github.com/AFD-Illinois/ipole (BSD-3)
- Comparaison codes polarisés EHT — https://arxiv.org/html/2303.12004
