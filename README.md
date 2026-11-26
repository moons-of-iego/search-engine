# Moteur de recherche dans la littérature scientifique

> **MOD 7.2 : Introduction à la science des données** (BE séances 4, 5 et 6)  
> **Groupe**: Raphaël Davin, Maxence Dordenart, Mathis Frinzi

---

## Présentation du sujet

Ce projet consiste à concevoir un **moteur de recherche sémantique** appliqué à un corpus de plus de 25 000 publications scientifiques.

Étant donnée une publication requête $q$, l'algorithme doit attribuer un score d'appariement $S(q, c)$ à un ensemble d'articles candidats $C$ composé de :

- **5 articles cités** par la publication requête (exemples positifs $C^+$) ;
- **~25 articles aléatoires** issus du corpus (exemples négatifs $C^-$).

L'objectif est de faire remonter préférentiellement les 5 citations pertinentes en tête de classement, en explorant progressivement plusieurs familles de représentations (**approches creuses**, **représentations denses** et **information de structure** via le graphe de citations).

---

## Fonctionnement du repo

La branche `maxence` contient le code développé par Maxence (logique) (actuellement jusqu'à la partie 3 du BE avant le test des variantes).  
Pour pousser du code, poussez une branche locale et on se mettra ensuite d'accord ensemble sur ce qu'on met dans le main qui sera le rendu final.

---
