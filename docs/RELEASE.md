# Maintainer release procedure

Source is kept in Git; frozen paper artifacts are kept in a separate Hugging Face
model repository. Do not upload a populated research folder or generated datasets
to the source repository.

Stage existing results without training or uploading:

```powershell
python -m stellar_workflow.release --source-root PATH_TO_ORIGINAL_RESEARCH --output-dir outputs/release/paper-v1
python -m stellar_workflow reproduce-paper --bundle outputs/release/paper-v1 --run-dir outputs/reference/local-check
```

The staging command verifies binary-copy hashes, retains original tuner records,
creates portable metadata and a checksum manifest, and records source hashes. It
does not infer an unknown original training commit. It requires a fresh directory.
Check the model card, license, manifest and local reproduction before publication.

Publication is a separate authorized action. Sign in locally with `hf auth login`;
never paste a token into an issue, notebook, source file or chat. After authorization,
create the public **model** repository BrunoBarreto/sdss_dr12_stars_regression and
upload only the staged release. Create the paper-v1 tag and record the returned
40-character immutable commit in reference/artifacts.json, preserving the staged
manifest SHA-256. The dataset repository is separate and must not be overwritten.

Before releasing the source, anonymously fetch the pinned model revision into an
empty location, verify every manifest entry, and run reproduce-paper without
--bundle. Preserve the published commit and release tag for future reproductions.

The published paper-v1 release is pinned to
3a7b0b12020f577c49d806cd930979f3d5117c27. Its manifest SHA-256 is
327579006124005a26104a218e33a29319f41fe68a6c4e99627db732ea80ebc2.
Fresh training, public release downloads and local --bundle reproduction do not
depend on signing in. Future publication requires separate authorization; source
commits and pushes are never automatic.
