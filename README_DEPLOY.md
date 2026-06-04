# Deployment

Deployment is completely automated and created on top of dockerhub and github actions.

Process step by step:

1. new tag `git tag -a "v...` published to git repository
1. github actions executed:
    - build frontend image
    - push frontend image to dockerhub
    - build api image
    - push api image to dockerhub
    - connect to production server
    - upload some repository artifacts to server
    - pull docker images
    - restart containers
    - cleanup