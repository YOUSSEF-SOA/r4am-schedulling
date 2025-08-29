# Ordonnancement des opérations de maintenance avec Reinforcement Learning

## Table des matières
- [Introduction](#introduction)
- [Bibliothèques utilisées](#bibliotheques)
  - [Stable-Baselines3](#stable-baselines3)   
  - [Gymnasium](#gymnasium)
- [Utilisation](#utilisation)  
  - [1. Installation des dépendances](#1-installation-des-dépendances)  
  
- [Environnement : MaintenanceSchedulingEnv](#environnement)
  - [Action space](#action-space)
  - [Observation space](#observation-space)
- [Fonctions principales de l’environnement](#fonctions)
  - [reset()](#reset)
  - [get_current_state_representation()](#get-current-state-representation)
  - [does_overlap()](#does-overlap)
  - [is_time_slot_free()](#is-time-slot-free)
  - [update_observations_for_legal_actions()](#update-observations)
  - [increase_time_step_logic()](#increase-time-step-logic)
  - [increase_time_step()](#increase-time-step)
  - [step()](#step)
  - [is_done()](#is-done)
  - [get_action_mask()](#get-action-mask)

---

## Introduction <a id="introduction"></a>

Le code réalise l’ordonnancement des opérations de maintenance, avec pour objectif d’optimiser le temps de travail des techniciens.  

Pour cela, un algorithme de **Reinforcement Learning (RL)** est utilisé comme outil d’optimisation.  
L’utilisation du RL nécessite deux éléments principaux :  

1. Un algorithme de RL  
2. Un environnement dans lequel l’agent peut explorer les différentes possibilités de planification  

---

## Bibliothèques utilisées <a id="bibliotheques"></a>

### Stable-Baselines3 <a id="stable-baselines3"></a>

**Stable-Baselines3** est une bibliothèque Python qui fournit des implémentations fiables et modernes des principaux algorithmes de RL.  
Elle propose plusieurs types d’algorithmes basés sur les principes du RL.  

Dans ce contexte, nous avons choisi **PPO (Proximal Policy Optimization)**.  

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

def mask_fn(env):
    return env._get_action_mask()

env = MaintenanceSchedulingEnv()

# Entraînement
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```
---  

### Gymnasium <a id="gymnasium"></a>  

**gymnasium**  est une bibliothèque open-source pour la création et l’interaction avec des environnements de Reinforcement Learning (RL).  

Elle permet de définir :  
- l’**action space** (ensemble des actions possibles que l’agent peut entreprendre),  
- l’**observation space** (états observables que l’agent peut recevoir de l’environnement),  
- la **logique de simulation** (récompenses, transitions d’état, conditions de fin d’épisode).  

Dans ce projet, **Gymnasium** est utilisé pour :  
- définir l’environnement `MaintenanceSchedulingEnv`,  
- structurer les actions et observations pour être compatibles avec l’algorithme PPO,  
- fournir une interface standardisée pour l’entraînement et le test du modèle RL.  

---

#### Données d’entrée de l’environnement

L’environnement utilise plusieurs ensembles de données :  

- **Techniciens** :  
  - `id` : identifiant du technicien  
  - `compétences` : domaines d’expertise  
  - `disponibilité` : créneaux horaires disponibles  

- **Machines** :  
  - `id` : identifiant unique  
  - `disponibilité` : état (en marche ou en maintenance)  

- **Workorders (ordres de travail)** :  
  - `id` : identifiant de l’ordre  
  - `machine_id` : identifiant de la machine concernée  
  - `compétences nécessaires` : prérequis techniques  
  - `priorité` : niveau d’urgence  
  - `durée de traitement` : temps requis pour finaliser la tâche  

---

#### Intégration avec PPO

Grâce à Gymnasium, l’environnement est :  
- **compatible avec Maskable PPO**,  
- **facilement testable et réinitialisable**,  
- **standardisé** : les fonctions `reset()` et `step(action)` suivent la même interface que les autres environnements RL, facilitant l’entraînement avec Stable-Baselines3.




---

## Utilisation <a id="utilisation"></a>
Prérequis
- Python 3.11
- pip installé
- (Optionnel mais recommandé) Créer un environnement virtuel pour isoler les dépendances
### 1. Installation des dépendances  
Clonez le dépôt et installez les bibliothèques nécessaires :  
```python
git clone https://github.com/YOUSSEF-SOA/r4am-schedulling.git
cd r4am-schedulling
pip install -r requirements.txt
```

---
## Environnement : MaintenanceSchedulingEnv  <a id="environnement"></a>

#### Action space  
l’espace d’action doit permettre à l’algorithme Maskable PPO de sélectionner un couple (workorder_id, technician_id). Pour résoudre le problème de Maskable PPO qui ne supporte pas un masque 2D, on encode les actions comme suit :   
```python
-action space: self . action_space = spaces . Discrete ( self . nombre_des_actions ) wo_id = action // nombre_des_techniciens tech_id = action % nombre_des_techniciens

```
#### Observation space

représente l’ensemble des informations accessibles à l’agent. 
```python
self . observation_space = spaces . Dict ({ ’action_mask ’: gym . spaces . Box (0 , 1 , shape = ( self . nombre_des_actions ,) , dtype = bool ) , 
’wo_obs ’: gym . spaces . Box ( low =0.0 , high =1.0 , shape =( self . nombre_des_wo , 6) , dtype = float ) , 
’tech_obs ’: gym . spaces . Box ( low =0.0 , high =1.0 , shape =( self . nombre_des_tech , 3) , dtype = float) ,8 })
```
— action_mask : indique quelles actions sont autorisées à chaque étape pour Maskable PPO.  
— wo_obs : matrice représentant l’état des ordres de travail. Chaque ligne contient 6 colonnes :    
1. Ordre de travail éligible (conditions : non exécuté, machine et technicien disponibles),
2.  temps restant pour un ordre en cours,
3.  durée déjà exécutée pour un ordre en cours,
4.  Durée d’inactivité d’un ordre de travail éligibe entre deux pas de temps successif .
5.  Cumul des durées d’inactivité
6.  Durée d’exécution de chaque ordre de travail
— tech_obs : matrice représentant l’état des techniciens (3 colonnes) :
 1. technicien éligible,  
 2.  temps restant pour finir une tâche en cours,  
 3 . durée déjà exécutée par le technicien.  
---
## Fonctions principales de l’environnement <a id="fonctions"></a>

### reset() <a id="reset"></a>
Réinitialise l’environnement au début d’un nouvel épisode.  
Remet à zéro les techniciens, les machines et les ordres de travail.  

### get_current_state_representation() <a id="get-current-state-representation"></a>
Renvoie une représentation vectorielle de l’état actuel.  
Sert d’entrée au modèle de RL.  

### does_overlap(interval1, interval2) <a id="does-overlap"></a>
— Ce qu’elle fait : Vérifie si deux intervalles de temps se chevauchent. — Exemple concret :   
— Disponibilité M1 : [10h, 12h]   
— Occupation M1 : [11h, 13h] — max(10, 11) = 11, min(12, 13) = 12 ⇒ 11 < 12 → chevauchement.  

### is_time_slot_free(resource_intervals, slot_start, slot_end) <a id="is-time-slot-free"></a>
Vérifie si une ressource est libre dans un intervalle donné.  
— Exemple concret :  
Technicien occupé sur [(8h, 10h), (13h, 15h)].  
On veut planifier [9h, 11h] ⇒ chevauchement avec (8h, 10h) ⇒ False.

### update_observations_for_legal_actions() <a id="update-observations"></a>
Met à jour l’éligibilité des WO et des techniciens (compétences, disponibilités), et régénère le masque d’actions légales. Appelée après chaque action ou avancement de temps.

### increase_time_step_logic(delta_time_minutes) <a id="increase-time-step-logic"></a>  
Incrémente le temps (en minutes) et met à jour les variables dépendantes du temps : WO en cours, temps techniciens, inactivité des WO.

### increase_time_step() <a id="increase-time-step"></a>
Avance l’horloge jusqu’au prochain événement dans self.next_event_queue.  
— Met à jour les temps écoulés, nettoie les intervalles passés, régénère les observations et applique une récompense négative pour l’inactivité.  

### step(action) <a id="step"></a>
Exécute une action choisie par l’agent.  
Simule un pas de temps où l’agent choisit une action (affectation d’un WO ou no-op).  
 — Initialisation : — reward = 0, done = False, truncated = False, info = {}.   
 — Gestion d’une no-op : L’action no-op joue un rôle essentiel dans cet environnement. Elle permet à l’agent de progresser dans le temps jusqu’au prochain événement (comme la disponibilité d’une ressource).  
 — Action d’affectation : Lorsque l’agent choisit une action d’affectation (un WO à un technicien), la fonction step exécute une série d’opérations critiques :  
 — Identification du WO et du technicien (à partir "action").   
 — Calcul début/fin de tâche, mise à jour ressources (technicien, machine).  
 — Ajout dans next_event_queue, mise à jour statut WO et journalisation dans solution_log.  
— Une récompense positive, proportionnelle à la durée d'exécution de l'ordre de travail choisi, est attribuée afin de favoriser la maximisation du temps de travail des techniciens.  
— Vérifications finales :   
— Si toutes les tâches sont accomplies ou s’il n’y a pas d’événements futurs, et que le nombre d’ordres légaux est égal à 0 ⇒ done = True. — Retourne : état actuel, récompense (scalée), done, truncated, et info.    

### is_done() <a id="is-done"></a>
Vérifie si l’épisode est terminé.  

### get_action_mask() <a id="get-action-mask"></a>
Renvoie un masque indiquant quelles actions sont légales et lesquelles doivent être bloquées.  


