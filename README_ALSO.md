# Compte rendu exam Prometheus et Grafana

Dans ce compte rendu, nous listerons les quelques commandes de prise en main du projet pour son évaluation et revisiterons le provisioning d'alertes et de dashboard.

## Les quelques commandes utiles du projet

`make all` :: pour lancer tout les dockers
normalement, sur un pc/vm vierge, tous les éléments du "provisioning" seront disponibles à nouveau pour évaluation.

J'ai testé sur une autre machine. Attention toutefois à des dockers containers grafana d'autres projets, un `docker container prune` peut être due - voire un remove.

`make stop` :: tout arréter

`make traffic` :: pour générer du traffic sur le endpoint `/predict`. génération aléatoire de 1 à 60 predicts par paquets de 10.

`make evaluate` :: build et lance un evaluate pour la correction.

`http://youhost:8080/`  :: pour accéder au swagger de l'API
- on peut faire un /predict à la main
- on peut faire un /train
- on peut faire un /evaluate en ne renseigant que la période, parmi "week{1,2,3}_february", on peut laisser le champ data vide `{}`
- on peut faire aussi un /metrics suivant les dashboards qu'on veut regarder

`docker logs -f container` :: pour les intimes ragards dans les logs

Recommandaton d'usage :
- dézip ou clone du projet sur la vm
- à la racine, `make all`
- ouverture du swagger http://liora-vm-77gi:8080/docs
- ouverture de prometheus http://liora-vm-77gi:9090/prometheus
- ouverture de grafana http://liora-vm-77gi:3000/




## Exploration de grafana et stratégie de mise à jour
Autant les premiers graphiques ou configuration grafana sont à peu près intuitives, en allant chercher des métriques prometheus dans sa datasource, autant on ne soupconne pas les exercices d'équilibriste pour échanger ou transporter des définitions graphiques entre environement. Aussi le concepte de "dashboard as code" arrive relativement vite, avec une spécification complète du modèle objet des assets graphiques en json ou en yaml.

La lecture des fichiers ainsi obtenus avec le déploiment en profondeur de tout le modèle objet reste profondément inhumain, en dépit de sa simplicité et de sa rationalité, sa longeur de spécification rend la maintenance de ces documents fastidieuse.

Une première stratégie de mise à jour simple est la possibilité de mettre à jour et de maintenir les graphiques et configuration dans l'interface en wysiwyg, puis de faire des exports ou des imports des "dashboards as code" ainsi générés. Le modèle objet n'est pas des plus claire et j'ai mis du temps avant de trouver comment mettre deux labels sur les axes d'un chart X Y. Les genIA du quartier propose allégrement des longues générations de dashboard en json (v1 ou v2 on ne sait pas trop) assez facilement.

Ensuite pour la distribution de nos configurations, le fameux "dashboard/alert provisioning" entre en jeu.

## Le provisioning des dashboards et des alertes
Le dossier *provisioning* contient donc en quelque sort les templates de dashboards ou d'alertes que l'on peut déployer sur un nouvel environement.

Il ne suffit pas de mettre à jour les json pour voir un changement dans l'interface, et un *graceful restart* ni fera rien non plus, car les config provisionnés sont chargées en base une seule fois au premier démarrage de l'application. Il nous faudra trouver un truc officiel pour ce chargement.

Au moment du provisioning, les .json définis dans `deployment/grafana/provisioning/dashboards/*.json`
vont se charger depuis le disque dans la base de données de grafana. Ces dashboards provisionnés ne peuvent pas être modifiés facilement dans l'interface, on ne peut pas faire **`save`**. Il faut faire un `save as` seulement
puis un *export* du dashboard complet afin de cycler sur une mise à jour. Bref des qu'on commence le provisioning on entre dans une autre approche.

## Export du dashboard grafana
l'*export du dashboard* par défault se fera en version `v2` du dashboard, reconnaissable avec un markeur **"kind:"** dans les balises json. Et cet export sera malheureusement complétement inutilisable en provisioning. Déconvenue !

Pour exporter en `v1`, qui est la seule version compatible avec le provisioning, il faut aller chercher l'option `"classic"` lors de l'export du dashboard. Il y a des risques de pertes en ligne de la `v2` à la `v1`, notamment sur certaine mise en formes avancées commes les onglets.

Ensuite, une fois le code affiché en version `v1`, soit on le passe par le clipboard dans un `vi` ouvert à coté, ou bien on save et on transporte le fichier de son disque local, au disque du serveur de l'api.

|| *Rappel, pour supprimer 999 lignes dans* `vi`*, on fait* `d 9 9 9 d`

## le reload du provisioning des dashboards ou alertes
Ensuite pour remettre le json sauvegarder sur le dossier de provisioning dans la base,
il est possible de lancer un reload. Pour cette instruction, on appel une api grafana avec une authentification !!

Bon, allons chercher son password dans son navigateur, puis lancer un `read -s`, certe on trimbale son password dans le clipboard, mais au moins il ne trainera pas dans l'historique des commandes linux ou les logs puppets.

`read -s passwd`

Puis un curl reload de grafana dédié au rechargement des json dans la base de données.

`curl -X POST http://admin:${passwd}@localhost:3000/api/admin/provisioning/dashboards/reload`

Si on redémarre sur une autre machine, on se rend compte de l'importance du provisioning et de son corollaire de lourdeur administrative. En effet, les dashboards provisionnés sont read-only. Pour modifier, il faut passer par une copie puis la faire syndiquer.

Par exemple ici, au redémarrage, il manquera encore - au début de la session de check - la bonne datasource pour un provisioning des alertes.

Voyons ensemble comment approcher la situation.

## Exemple de correction de datasource d'alertes

D'abord à l'ouverture nous sommes confrontés à un message `failed to build query 'A': data source not found` dans l'alerte ApiPredictLatencyHigh

C'est le signe d'un identifiant de datasource qui n'est plus disponible sur le nouveau host. A l'instar des dashboards, voyons si dans les Alertes il y a aussi un uid de datasource à spécifier dans nos fichiers yaml.

-> Effectivement, au controle nous avons aussi  `datasourceUid: cfobsja411b7kf` - qui est un indentifiant hérité de la première configuration.

Pour rappel, la configuration d'un `uid: prometheusuid00` dans notre dashboard.yaml permet d'avoir un controle transversal machine de l'indentification de la source prometheus.

*NB: contrairement aux nombreuses suggestions du cours et des IA, seule la spécification compléte http://prometheus:9090/prometheus permet d'avoir un fonctionnement nominal de l'application.*

## Première correction de la datasource 

```
# diagnostique :

 ==> grep -nR cfobsja411b7kf deployment/grafana/provisioning
deployment/grafana/provisioning/alerting/bike-api-support-L2.yaml:16:              datasourceUid: cfobsja411b7kf
deployment/grafana/provisioning/alerting/bike-api-support-L2.yaml:76:              datasourceUid: cfobsja411b7kf


# remplacement

 ==> (cd deployment/grafana/provisioning/alerting ; sed -i 's:cfobsja411b7kf:prometheusuid00:g' *.json *.yaml *.yml )
sed: can't read *.json: No such file or directory
sed: can't read *.yml: No such file or directory

# restarting services is due
make all

# reload en alerting ?
curl -X POST http://admin:${passwd}@localhost:3000/api/admin/provisioning/alerting/reload

 ==> curl -X POST http://admin:${passwd}@localhost:3000/api/admin/provisioning/alerting/reload
{"message":"Alerting config reloaded"}

# transfert sur autre serveur

```

## Transfert du correctif par git (mais on fait comme on veut)
Puis on passe par git pour mettre à jour le repo, et aller sur un autre serveur pour tester.

```
ubuntu@ip-172-31-35-255:~/PromGraf-MLOps-Exam-Student
 ==> git add .

ubuntu@ip-172-31-35-255:~/PromGraf-MLOps-Exam-Student
 ==> git status

On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   README_ALSO.md
        modified:   deployment/grafana/provisioning/alerting/bike-api-support-L2.yaml


ubuntu@ip-172-31-35-255:~/PromGraf-MLOps-Exam-Student
 ==> git commit -m "update alert datasource"

[main feed642] update alert datasource
 Committer: schmilblick-ai <ubuntu@ip-172-31-35-255.eu-west-1.compute.internal>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 2 files changed, 2 insertions(+), 2 deletions(-)
 create mode 100644 README_ALSO.md

ubuntu@ip-172-31-35-255:~/PromGraf-MLOps-Exam-Student
 ==> git push

Enumerating objects: 13, done.
Counting objects: 100% (13/13), done.
Delta compression using up to 2 threads
Compressing objects: 100% (7/7), done.
Writing objects: 100% (7/7), 764 bytes | 764.00 KiB/s, done.
Total 7 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
To github.com:schmilblick-ai/PromGraf-MLOps-Exam-Student.git
   4beba0c..feed642  main -> main


```
## Portage incrémental d'une modification d'alerte
puis sur l'autre serveur - normalement on redémarre tout sans encombre, mais sinon incrémentalement ca donne `pull + reload` :

```
# on suppose les services sont up (make all)

ubuntu@ip-172-31-37-17:~/PromGraf-MLOps-Exam-Student$
==> git pull

remote: Enumerating objects: 13, done.
remote: Counting objects: 100% (13/13), done.
remote: Compressing objects: 100% (5/5), done.
remote: Total 7 (delta 2), reused 7 (delta 2), pack-reused 0 (from 0)
Unpacking objects: 100% (7/7), 744 bytes | 372.00 KiB/s, done.
From github.com:schmilblick-ai/PromGraf-MLOps-Exam-Student
   4beba0c..feed642  main       -> origin/main
Updating 4beba0c..feed642
Fast-forward
 README_ALSO.md                                                    | 0
 deployment/grafana/provisioning/alerting/bike-api-support-L2.yaml | 4 ++--
 2 files changed, 2 insertions(+), 2 deletions(-)
 create mode 100644 README_ALSO.md

# Puis des jsons de provisioning, on passe en base grafana
ubuntu@ip-172-31-37-17:~/PromGraf-MLOps-Exam-Student$ 
==> curl -X POST http://admin:${passwd}@localhost:3000/api/admin/provisioning/alerting/reload

{"message":"Alerting config reloaded"}
```

et enfin, en refresh de la page, les alertes retrouvent leur statuts et leur datasources.


## Merci pour votre attention,