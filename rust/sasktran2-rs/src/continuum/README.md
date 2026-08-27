# MT_CKD 4.3 spectral data

The pure-Rust evaluator intentionally contains no AER coefficient data. On
first use, the Python interface downloads the separately licensed table through
`StandardDatabase` using this key:

`continuum/mt_ckd_4_3.bin`

The server copy is staged at
`/Volumes/SasktranFiles/sasktran2_db/v_latest/continuum/mt_ckd_4_3.bin` and is
served from the corresponding `arg.usask.ca/sasktranfiles` URL. It contains 22
little-endian `f64` coefficient fields generated from AER-RC/MT_CKD tag `4.3`
(commit `1dad6e29363a2dc75d0eb642b7321b5db51079cc`) and its pinned LBLRTM source.
Its SHA-256 is:

`06b5600ecb0f5a3417c46d226555bc516f5a55ae93b19b6e7b5431e50f8f0459`

The upstream double-precision validation spectra are stored only on the data
server as `continuum/mt_ckd_4_3_reference.bin`, with SHA-256:

`0fdc39229ae020f2979511acfc81600ddfaa0b9bc0bbb03bb86a5ce3d01a9760`

The AER terms are reproduced in `MT_CKD_LICENSE.txt`, printed before the first
coefficient download, and downloaded beside the cached table. The coefficient
and validation data are not covered by SASKTRAN2's MIT license.

## Evaluator convention

The Rust and Python APIs return 1,991 extinction-coefficient samples in m^-1
on the native 0 to 19,900 cm^-1 grid at 10 cm^-1 spacing. Internally MT_CKD is
evaluated for a fixed one-metre column, making its optical depth numerically
equal to extinction in m^-1. Geometric path length is therefore deliberately
not part of the continuum API.

The scalar evaluator uses ordinary `f64` arithmetic. The linearized evaluator
uses the same generic kernel with forward-mode dual numbers for pressure,
temperature, H2O VMR, CO2 VMR, and O3 VMR.
