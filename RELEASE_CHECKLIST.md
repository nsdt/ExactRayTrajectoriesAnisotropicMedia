# Release checklist

- [ ] Confirm that Yohei Nishidate is the intended author/copyright holder and MIT is the intended license.
- [ ] Run `make setup` in a clean checkout.
- [ ] Run `make verify`; confirm all assertions pass and the CSV files regenerate without changes.
- [ ] Confirm `results/numerical-experiments.json` records the release environment.
- [ ] Confirm the package contains no manuscript, PDF, Overleaf output, temporary file, cache, build artifact, old project, or GRIN13 file.
- [ ] Choose the GitHub owner and repository name.
- [ ] Add the final repository URL to README.md and `repository-code` to CITATION.cff.
- [ ] Create the public release and archive it with Zenodo.
- [ ] Add the assigned Zenodo DOI to README.md and CITATION.cff; add related identifiers to .zenodo.json if desired.
- [ ] Validate CITATION.cff and .zenodo.json using the services' current validators.
- [ ] Tag the exact archived revision as `v1.0.0` (or update all version fields consistently).
