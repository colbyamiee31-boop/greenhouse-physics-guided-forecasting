# GitHub + Zenodo release guide

## Recommended architecture

### GitHub
Publish source code, configuration, documentation, metadata, and lightweight reference outputs.

Recommended repository name:
`greenhouse-physics-guided-forecasting`

### Zenodo
Archive the complete reproducibility ZIP containing the processed v3 data and frozen result files.

This separation keeps the Git repository lightweight while giving reviewers a permanent,
versioned data/software archive.

## GitHub steps
1. Create a new repository named `greenhouse-physics-guided-forecasting`.
2. Do not initialize it with conflicting README/license files if you plan to upload the prepared bundle.
3. Upload/push the contents of `GITHUB_REPO_READY_v1.zip`.
4. Confirm the root contains `CITATION.cff` and `.zenodo.json`.
5. Create a GitHub release/tag, recommended: `v2.0.0`.

## Zenodo via GitHub integration
1. Sign in to Zenodo and link the GitHub account.
2. Open the GitHub integration page and sync repositories.
3. Enable the new repository.
4. Create the GitHub `v2.0.0` release.
5. Check the automatically created Zenodo record before using its DOI in the manuscript.

Because `.zenodo.json` and `CITATION.cff` are both present, Zenodo uses `.zenodo.json`
for GitHub-release metadata.

## Alternative: manual Zenodo deposit
Upload `greenhouse_physics_forecasting_ARCHIVE_READY_v3.zip` directly to Zenodo,
fill in the prepared metadata, review access/license settings, and publish only after all
authors approve public data release.

## Do not publish until
- all authors approve the chosen software license;
- institutional/data-sharing permission for the processed greenhouse records is confirmed;
- metadata and author order are checked;
- the canonical forcing point is visibly 40.54°N, 81.30°E.
