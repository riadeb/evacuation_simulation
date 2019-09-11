import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as anim
import time
from xlrd import open_workbook
import pandas as pd



haut = 5.01  # hauteur du tableau #Nombre de cellule
larg = 10.01  # largeur du tableau
mur = 8  # cÃ´tÃ© d'une cellule
interval = 0.01 #secondes
largeurporte = 1
##
class individu:
    def __init__(self,x,y,vitesse,angle):
        self.x = x
        self.y = y
        self.vitesse = vitesse
        self.angle = angle
        line, = ax.plot(self.x,self.y,marker='o', color='r',lw=5)
        self.line = line
        self.nbrchoc = 0
    def affichage(self):
        self.line.set_data(self.x,self.y)
    def deplacement(self):
        self.x += interval*self.vitesse*np.cos(self.angle)
        self.y += interval*self.vitesse*np.sin(self.angle)
    def choc(ind1,ind2):
        d = np.sqrt((ind1.x-ind2.x)**2 + (ind1.y-ind2.y)**2 )
        if d < 0.1: 
            ind1.nbrchoc += 1
            ind2.nbrchoc += 1
            return True
        else: return False
    def rebond(self,ind2):
        v1 = np.array([self.vitesse*np.cos(self.angle),self.vitesse*np.sin(self.angle)])
        v2 = np.array([ind2.vitesse*np.cos(ind2.angle),ind2.vitesse*np.sin(ind2.angle)])
        u = np.array([ind2.x - self.x,ind2.y - self.y])
        v1nouv = v1 - np.vdot(v1-v2,u)*u
        v2nouv = v2 + np.vdot(v1-v2,u)*u
        self.angle = np.arctan2(v1nouv[1],v1nouv[0])
        ind2.angle = np.arctan2(v2nouv[1],v2nouv[0])
        tang = np.arctan2(ind2.y-self.y,ind2.x - self.x)
        angle = 0.5 * np.pi + tang
        self.x += 0.1*np.sin(angle)
        self.y -= 0.1*np.cos(angle)
        ind2.x -= 0.1*np.sin(angle)
        ind2.y += 0.1*np.cos(angle)

##
nbrdivx = 500
nbrdivy = 500
obstaclenum = 500 #Valeur des zones ou il ya obstacles
pasx = larg/nbrdivx
pasy = haut/nbrdivy
maillage = np.zeros((nbrdivx,nbrdivy))
maillage.fill(obstaclenum)
def convertirencoef(x,y): #Convertir coordonnee en i et j
    i = int(x*nbrdivx/larg)
    j = int(y*nbrdivy/haut)
    return i,j
obstacles = list()
def siobstacle(iv,jv):
    siobstacle = False
    for ob1,ob2 in obstacles:
        i1,i2 = ob1[0],ob2[0]
        j1,j2 = ob1[1],ob2[1]
        if iv >= i1 and iv <= i2 and jv >= j1 and jv <= j2: 
            siobstacle = True
            break
    return siobstacle
##Mur
obstacles += [[convertirencoef(mur,2.2),convertirencoef(mur,haut)]] #liste dobstacles du debut a la fin
obstacles += [[convertirencoef(mur,0),convertirencoef(mur,2)]]
obstacles += [[convertirencoef(mur+i,0),convertirencoef(mur+i,2)] for i in np.linspace(-0.1,0.1,50)]
obstacles += [[convertirencoef(mur+i,2.2),convertirencoef(mur+i,haut)] for i in np.linspace(-0.1,0.1,50)]

### Obstacle Horizontale
#obstacles += [[convertirencoef(2,4),convertirencoef(5,4)]]
obstacles += [[convertirencoef(2,4+i),convertirencoef(5,4+i)] for i in np.linspace(-0.2,0.2,50)]
### Obstacle Verticale 
#obstacles += [[convertirencoef(5,1),convertirencoef(5,4)]]
obstacles += [[convertirencoef(5+i,1),convertirencoef(5+i,4)] for i in np.linspace(-0.1,0.1,50)]

##### calcul de la distance - Programmation dynamique
file = list()
buts = [convertirencoef(mur,2),convertirencoef(mur,2.2)]
point1,point2 = buts[0],buts[1] #Extremites du mur
for i in range(point1[0],point2[0]+1):
    for j in range(point1[1],point2[1]+1):
        maillage[i,j] = 0 #Le but recoit la valeur 0
        file.append((i,j))

while len(file) > 0:
    i,j = file.pop() #Point actuel
    voisins = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
    for iv,jv in voisins:
        if iv >= nbrdivx or jv >= nbrdivy or iv < 0 or jv < 0: continue
        if siobstacle(iv,jv) : continue
        nouvellevaleur = maillage[i,j] + 1
        if nouvellevaleur < maillage[iv,jv]: 
            maillage[iv,jv] = nouvellevaleur
            file.insert(0,(iv,jv))
np.savetxt("file",maillage,delimiter=" ; ",newline="\n")

### Calcul de langle que fait le vecteur gradient 
gradient = np.zeros_like(maillage)
for i in range(nbrdivx):
    for j in range(nbrdivy):
        gauche = i-1 if i > 0 else i
        droite = i+1 if i < nbrdivx -1 else i
        dessus = j+1 if j < nbrdivy -1 else j
        dessous = j-1 if j > 0 else j
        
        y = maillage[i,dessus]-maillage[i,dessous]
        x = maillage[droite,j]-maillage[gauche,j]
        x /= pasx
        y /= pasy
        if x != 0:
            gradient[i,j] = np.arctan(y/x)
        elif y ==0: gradient[i,j] = 0
        else: gradient[i,j] = np.pi*np.sign(y)
#####
# INITIALISATION
fig,ax = plt.subplots(figsize=(13,7));
ax.axis([0,larg,0,haut])
ax.set_xticks([])
ax.set_yticks([])
#Mur
ax.plot([mur,mur],[0,2],color="blue") 
ax.plot([mur,mur],[haut,2.2],color="blue") 

#obstacles1
ax.plot([5,5],[4,1],color="blue")
ax.plot([2,5],[4,4],color="blue")
# Creation des individus
Lesindividus = []
nbrindividus = 100
probabilite = 0.001
N = 0
while N < nbrindividus:
    for x in np.linspace(0,mur-0.1,nbrdivx):
        for y in np.linspace(0,haut-0.1,nbrdivy):
            p = np.random.random()
            if p <= probabilite and N < nbrindividus:
                Lesindividus.append(individu(x,y,5,1.2))
                N += 1
Lesindividus.append(individu(4,2.9,5,1.2))
#Animation
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
choc_text = ax.text(0.5, 0.95, '', transform=ax.transAxes)
N = 0
def simaximum(i,j):
    voisins = [(i-2,j),(i+2,j),(i,j-2),(i,j+2)]
    return maillage[i,j] >= max([maillage[iv,jv] for iv,jv in voisins])
def miseajour(sec):
    global N
    k = sum([i.nbrchoc for i in Lesindividus])
    N = k if k > N else N
    choc_text.set_text("Chocs : " + str(N//2)) 
    for indice,ind in enumerate(Lesindividus):
        x,y = ind.x,ind.y
        i,j = convertirencoef(x,y)
        if i >= nbrdivx or j >= nbrdivy: 
            ind.x = larg + 1
            ind.affichage()
            Lesindividus.remove(ind)
            continue
        if i< convertirencoef(mur,0)[0] : time_text.set_text("Temps : "+str(sec)) #Si il ya tjrs un individu pas encore sorti
        ind.angle = gradient[i,j]
        for k in range(indice+1,len(Lesindividus)):
            ind2 = Lesindividus[k]
            if individu.choc(ind,ind2):
                ind.rebond(ind2)
               
        if i< convertirencoef(mur,0)[0] and j < nbrdivy - 3 and int(ind.angle) == 0 and simaximum(i,j) :
            ind.x += pasx
            ind.y +=  pasy
            
        ind.deplacement()
        ind.affichage()
animation = anim.FuncAnimation(fig,miseajour,interval=10)
plt.show()
## Pour realiser un grand nombre de simulations et en stocker les resultats
def simulation():
    Lesindividus = []
    nbrindividus = 100
    probabilite = 0.001
    N = 0
    while N < nbrindividus:
        for x in np.linspace(0,mur-0.1,nbrdivx):
            for y in np.linspace(0,haut-0.1,nbrdivy):
                p = np.random.random()
                if p <= probabilite and N < nbrindividus:
                    Lesindividus.append(individu(x,y,5,1.2))
                    N += 1
    Lesindividus.append(individu(4,2.9,5,1.2))
    #Animation
    N = 0
    t0 = time.time()
    t = 0
    def simaximum(i,j):
        voisins = [(i-2,j),(i+2,j),(i,j-2),(i,j+2)]
        return maillage[i,j] >= max([maillage[iv,jv] for iv,jv in voisins])
    while len(Lesindividus) > 0:
        k = sum([i.nbrchoc for i in Lesindividus])
        N = k if k > N else N
        for indice,ind in enumerate(Lesindividus):
            x,y = ind.x,ind.y
            i,j = convertirencoef(x,y)
            if i >= nbrdivx or j >= nbrdivy: 
                ind.x = larg + 1
                ind.affichage()
                Lesindividus.remove(ind)
                continue
            if i< convertirencoef(mur,0)[0] : t = time.time() - t0 #Si il ya tjrs un individu pas encore sorti
            ind.angle = gradient[i,j]
            for k in range(indice+1,len(Lesindividus)):
                ind2 = Lesindividus[k]
                if individu.choc(ind,ind2):
                    ind.rebond(ind2)
                
            if i< convertirencoef(mur,0)[0] and j < nbrdivy - 3 and int(ind.angle) == 0 and simaximum(i,j) :
                ind.x += pasx
                ind.y +=  pasy
            
            ind.deplacement()
    return t,N//2
    
datasansobstacles = [simulation() for i in range(50)]
## Exportation des donness sous forme d'un fichier Excel
tempsmoye = sum([data[0] for data in datasansobstacles]) /len(datasansobstacles)
choc = sum([data[1] for data in datasansobstacles]) /len(datasansobstacles)
pd.DataFrame(datasansobstacles).to_excel('outputavecobs.xlsx', header=False, index=False)


