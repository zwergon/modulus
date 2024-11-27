# PINN with Modulus

Ce repository donne 2 petits exemples d'utilisation de modulus pour résoudre par l'utilisation de réseau PINN les deux problèmes physiques suivants:
 - Refroidissement d'une tasse de café (Utilisation de data et de l'équation de refroidissement )
 - Equations de Burgers ( que de la physique mais avec une approche paramétriques )


Ces exemples ont été testés sur une machine avec un GPU en utilisant docker:

```
docker run --user $(id -u):$(id -g) --rm --gpus all -v ~/repositories/modulus:/workspace nvcr.io/nvidia/modulus/modulus:24.04 python /workspace/coffee/coffee.py
```

Les résultats sont accessibles dans les notebooks:
- [Coffee](./notebooks/Coffee.ipynb)
- [Burger](./notebooks//Burger.ipynb)

Enjoy !