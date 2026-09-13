# Reference material and external implementation

The versioned paper HTML, text reading aids, conference PDF and
[provenance record](provenance.json) are retained with the publication.
[The source ledger](../notes/SOURCES.md) records their reading scope, exact
versions, hashes and access limitations.

The external Modular TTT checkout is excluded from this repository. Restore
it from the author-linked repository at the audited revision before running
scripts that load its definitions:

```sh
git clone https://github.com/ByteDance-Seed/Modular-TTT.git sources/Modular-TTT
git -C sources/Modular-TTT checkout --detach 33afe26100f8590272940e55dbee5067a8040da6
```

Run these commands from the study root only when the checkout is absent.
The existing local checkout has been retained unchanged. The numerical
audit records also hash the exact source files used by the scripts.

Model checkpoints under `models/`, scratch/download caches under `.cache/`,
Python environments and secrets are excluded from the Git publication.
Checkpoint verification requires the retained local model files (or a
separate reconstruction); the committed result records preserve their hashes.
