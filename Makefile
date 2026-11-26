# Thin mirror of tasks.py (the source of truth). Every target delegates to
# `uv run python tasks.py <target>`; parity is enforced by `tasks.py makefile-sync`.
RUN := uv run python tasks.py

.PHONY: help setup check eol hygiene size makefile-sync lint format format-check typecheck \
	test generate-demo run-pipeline report dashboard docker-build docker-smoke clean clean-data

help:
	@$(RUN) --help

setup:
	$(RUN) setup

check:
	$(RUN) check

eol:
	$(RUN) eol

hygiene:
	$(RUN) hygiene

size:
	$(RUN) size

makefile-sync:
	$(RUN) makefile-sync

lint:
	$(RUN) lint

format:
	$(RUN) format

format-check:
	$(RUN) format-check

typecheck:
	$(RUN) typecheck

test:
	$(RUN) test $(ARGS)

generate-demo:
	$(RUN) generate-demo $(ARGS)

run-pipeline:
	$(RUN) run-pipeline $(ARGS)

report:
	$(RUN) report $(ARGS)

dashboard:
	$(RUN) dashboard

docker-build:
	$(RUN) docker-build

docker-smoke:
	$(RUN) docker-smoke

clean:
	$(RUN) clean

clean-data:
	$(RUN) clean-data
