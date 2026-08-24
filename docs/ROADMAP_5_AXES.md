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

**S0 — Infrastructure & sortie HDF5**
Répertoire `grmhd/`, cible CMake séparée, structures de grille, écriture
HDF5 au format iharm. *Fait en premier volontairement* : pouvoir dumper et
inspecter l'état est l'outil de debug principal de tout ce qui suit.

**S1 — Métrique & coordonnées (Kerr-Schild)**
⚠️ Différence importante avec l'existant : le ray tracer utilise
Boyer-Lindquist (singulier à l'horizon). Un solveur GRMHD a besoin de
**Kerr-Schild** (pénétrant l'horizon) + transformation MKS/FMKS (maillage
log en r, concentration en θ vers le plan équatorial).
Livrables : `g_μν`, `g^μν`, `√-g`, symboles de Christoffel.
*Tests* : `g·g⁻¹ = I` à la précision machine ; `√-g` vs formule analytique ;
Christoffels vs différences finies de la métrique.

**S2 — Conversion variables primitives ↔ conservatives**
`P→U` : analytique, facile. `U→P` : **résolution non linéaire (Newton)** —
historiquement LE point fragile des codes GRMHD (référence : Noble et al.
2006). C'est ici que les codes échouent en pratique.
*Test* : aller-retour `P→U→P` sur un large domaine (facteur de Lorentz
élevé, forte magnétisation) avec récupération à la précision machine. Test
unitaire pur, aucune simulation nécessaire.

**S3 — Cœur hydro en RELATIVITÉ RESTREINTE d'abord (métrique plate)**
Reconstruction PLM (limiteurs minmod/MC), solveur de Riemann HLL,
intégrateur RK2/RK3 SSP.
*Pourquoi SRMHD avant GRMHD* : isole le cœur numérique de la complexité GR.
Si les chocs sont corrects en espace plat ET la métrique est correcte
séparément (S1), la GR n'ajoute essentiellement que des termes sources.
*Tests* : suite de tubes à choc SRMHD de Komissarov (solutions publiées).

**S4 — Termes sources GR → hydro GR (sans champ B)**
*Tests décisifs* :
- **Bondi/Michel** : accrétion sphérique stationnaire, **solution analytique
  exacte**. Le flot doit rester stationnaire. C'est le test le plus
  révélateur d'une erreur dans les termes sources ou la métrique.
- **Tore de Fishbone-Moncrief sans champ B** : équilibre exact, doit rester
  statique. Toute dérive = bug.

**S5 — Champ magnétique + transport contraint (CT)**
FLUX-CT (Tóth 2000) pour maintenir `∇·B = 0`. Sans ça, la simulation
diverge ou fabrique des monopôles magnétiques.
*Test* : `∇·B` reste au niveau de la précision machine sur des milliers de
pas. Invariant non négociable, à monitorer en permanence.

**S6 — Planchers, plafonds, conditions aux limites**
Planchers de densité/énergie interne, plafond de magnétisation σ et de
facteur de Lorentz ; sorties libres en r, axe polaire, périodicité en φ.
Ce n'est pas de la physique mais c'est **indispensable** : sans planchers,
NaN garantis dans la région du jet/funnel. Tous les codes du domaine en ont.

**S7 — Première vraie run : tore MRI**
Tore de Fishbone-Moncrief + boucle poloïdale faible → développement de la
turbulence MRI.
*Tests* :
- **Taux de croissance linéaire de la MRI vs prédiction analytique**
  (quantitatif, pas visuel).
- Puis métriques de **Porth et al. 2019** (comparaison de 9 codes :
  Athena++, BHAC, iharm3D, KORAL, H-AMR…) : taux d'accrétion, flux
  magnétique à l'horizon, séries temporelles. C'est la référence
  communautaire — le standard de validation le plus élevé accessible.

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
