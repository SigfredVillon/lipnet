.PHONY: help up
.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

up: ## Start all container
	docker compose up --build --detach

reset.env: ## Reset environment
	if [ -n "`docker ps -q`" ]; then docker stop `docker ps -a -q`; docker rm `docker ps -a -q`; fi
	if [ -n "`docker image ls -q`" ]; then docker rmi -f `docker image ls -q`; fi
	docker volume prune -f
	docker network prune -f
	docker system prune -f