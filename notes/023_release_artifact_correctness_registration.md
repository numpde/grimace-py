# Release Artifact Correctness Registration

`make test-package` validates two release shapes: the built wheel and the
source distribution. Each artifact is installed into a fresh environment and
then checked through `tests.run_installed_package_correctness`.

The writer support-count lane exposed a registration drift risk: a fixture
family and runtime test can be added correctly, while the installed-artifact
runner remains a manually maintained subset and misses the new release-relevant
test.

The immediate fix is to include
`tests.rdkit_serialization.test_writer_support_counts` in
`tests.run_installed_package_correctness`, so both wheel and sdist installs run
the count-evidence check.

The broader rule is that every release-significant evidence family needs one
explicit classification: exact parity, count evidence, known-gap diagnostic, or
source-only support data. Runners should be derived from that classification,
or at least checked against it, so future evidence lanes cannot silently miss
package-artifact validation.
