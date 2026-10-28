# LA library - notes for Claude

C++ linear algebra library (author: Jiri Pittner). This file collects the working rules, design
decisions and plans agreed with the author, so that they survive a fresh clone. Keep it ASCII, LF.

## Communication
- The author works in an ASCII xterm without UTF-8 fonts: replies in plain ASCII only, formulas in
  LaTeX style (A^{-1}, U^\dagger b, \sum_{k=1}^{N}, 10^{-15}, "-" for dashes).
- Do not ask for permission to run (non-destructive) shell commands; still confirm destructive or
  outward-facing actions (deleting user files, git push).
- Bug fixing workflow: locate the cause, explain it, propose the MOST EFFICIENT fix, WAIT for the
  author's OK, then patch, build, test, update documentation. One item at a time. Implement new
  methods only on request; design proposals: offer alternatives with a recommendation, the author picks.

## Build and test
- Build the library with `make -j 8` (after header changes most files rebuild; run it in the background).
- Flags used: `-std=gnu++11 -O3 -DDEBUG -DNO_STRASSEN -DFORTRAN_ -DNONCLAPACK`, libs `-lcblas -lblas -llapack`.
  Quick test program: `g++ -std=gnu++11 -I<repo> -O0 -DDEBUG -DNO_STRASSEN -DFORTRAN_ -DNONCLAPACK X.cc <repo>/.libs/libla.a -lcblas -lblas -llapack`
  (add `-fsanitize=address` for memory checks).
- Regression test of the optimizers: `make check` builds `testoptimize`; `./testoptimize` must report
  "123 passed, 0 failed" (exit code = number of failures). New optimizer tests go to testoptimize.cc;
  test problems live in testoptimize.h.
- `t` (t.cc) is a collection of if(0)/if(1) test blocks, most read n from stdin (`echo 10 | ./t`).
- NONCBLAS check: copy the repo with `cp -a` (keeps timestamps, otherwise automake reruns and fails),
  `make clean; make -j 8 CBLASOPT=-DNONCBLAS CBLASLIB=`.
- laerror() runs a shell (whoami, hostname) each time: loops raising thousands of laerrors are very slow.
- The shell is tcsh (`^!` in double quotes is history expansion; use bash -c or python for scripts).
- Manual: doc/manual.tex (LaTeX, 17 chapters, macros in doc/lamacros.tex: \code, \header, \lafunc in
  lareference, \latodo = yellow question box, \lawarning). Build with pdflatex three times (into a scratch
  directory with -output-directory), require 0 log lines starting with "!", copy manual.pdf into doc/.

## Coding rules
- Mirror the author's style in every file: tabs; braces on their own line indented with the body;
  no space after `if(`/`for(`; terse names; short `//` comments (Doxygen only where the file uses it);
  errors via laerror("..."); `#ifdef DEBUG` dimension checks; NOT_GPU/SAME_LOC like neighbouring code;
  noncblas.cc: follow the existing wrapper pattern (FORNAME, `#ifdef FORINT` copies).
- Byte-minimal edits; files are LF. If a CRLF file is found, report it (the author converts it), never
  convert line endings yourself. Edit scripts must work in binary mode.
- Templates declared in headers, explicit specializations (double, complex<double>) in .cc files;
  instantiations via `#define INSTANTIZE(T) ...` + `INSTANTIZE(double)`.
- No verbatim copying of foreign copyrighted code (GPLv3 with reference is OK).
- Always prefer the most efficient solution (no extra passes, copies or allocations), even if a wrapper
  elsewhere (e.g. noncblas.cc) must be extended.
- Default scope of optimizer work: optimize.h, optimize.cc, t.cc (+ testoptimize.*); ask before
  touching other files if that would give a better solution. Optimizer code must not call BLAS directly
  nor loop over vector/matrix elements - use the library classes.

## Copy-on-write (core design)
- All large-data classes share data on copy/assignment (NRVec, NRMat, NRSMat, SparseMat, SparseSMat,
  CSRMat, fourindex, Tensor (data, shape, names), polynomials, permutations, bitvector, *_from1).
- Every member function modifying the object as a whole must call copyonwrite() itself; a missing one
  is a library bug. resize() detaches.
- Element-level access is deliberately exempt (operator[], operator(), raw pointers, NRVec::swap(i,j),
  SparseMat/SparseSMat::add, bitvector bit operations): no copyonwrite() in inner loops; the caller calls
  copyonwrite() once before writing. Do not add reminder comments about this to the code.
- Also check user code (t.cc, test programs) for missing copyonwrite() before element writes; read
  possibly shared non-const objects through const references (DEBUG count check).

## Design decisions (2026-10)
- Complex NRSMat is HERMITIAN (lower triangle stored, upper = conj) in arithmetic: products,
  NRMat(NRSMat), NRSMat(NRMat) = (A+A^\dagger)/2, NRMat +=/-= NRSMat, sum(), fprintf(), linear_solve,
  diagonalize, trace2. BUT NRSMat::operator() (const AND non-const, also NRSMat_from1) is plain storage
  access without conjugation - a conjugating const operator() was tried and REVERTED as too dangerous;
  never reintroduce it. operator<<, permuted, submatrix(selection), rsum, csum keep storage semantics.
- SparseSMat is a general (also rectangular) row-wise sparse matrix; add(i,j,x) with default both=true
  adds x at (i,j) and conj(x) at (j,i); NRSMat(SparseSMat) takes the lower triangle; operator* is the
  general product A B (correct for hermitian, complex symmetric exp(iH) and general matrices);
  SparseSMat::gemm computes A^T B (A^\dagger B with 'c'); otimes_sparse = M + M^\dagger incl. doubled diagonal.
- SparseMat with the symmetric flag is complex symmetric (no conjugation); its permuteme_* members use
  0-based matrix indices, NRPerm/CyclePerm are always 1-based.
- Tensor(const NRSMat&) uses symmetry 2 (hermitian) for complex, 1 for real.
- a-X and a/X (scalar minus/divided by object) are deleted on purpose; a single integer constructor is
  forbidden for NRMat, NRMat_from1, Mat3 (Mat3(double) works).
- Real-storage gdiagonalize uses the dgeev convention (Re + i Im).
- The NRMat offset view constructor is a feature. MATPTR is kept (rarely used). CSRMat is unfinished and
  documented as unsupported, but kept. Strassen is experimental. CUDA (CUDALA) is deferred/experimental
  until the author has NVIDIA RTX hardware (known CUDA issues collected in suspected_bugs.md F10).

## Planned (only when the author asks)
- float version: float and complex<float> must ALWAYS use BLAS/LAPACK (s*/c* routines, noncblas wrappers)
  like double/complex<double>, never generic loops. INSTANTIZE(float) in optimize.cc currently needs float
  versions of diagonalize, linear_solve, singular_decomposition. BLAS-based NRVec members are meant for
  floating types only, not integers.
- NRSMat: possible bool member `hermitian` toggling hermitian vs complex symmetric behaviour
  (Tensor(const NRSMat&) must then respect it).
- SparseSMat: possible member `symmetry_type` (0 none, 1 symmetric, 2 hermitian) for NRSMat conversion,
  otimes_sparse, gemm, issymmetric, backward compatible.
- Optimizers (optimize.h/.cc, all 11 planned methods done: BFGS/DFP/SR1, L-BFGS, CG, Nelder-Mead,
  Powell, Adam/SGD, SPSA, GDIIS, QNG/Rotosolve, COBYLA, trust region/P-RFO, Levenberg-Marquardt):
  open for later - global optimization layer, further Hessian-based methods, FIRE, CMA-ES, Polyak averaging.

## Status files (may be absent in a fresh clone; the author keeps them outside git)
- suspected_bugs.md: verified bug list with SOLVED / NOT A BUG / DEFERRED marks and reasons; each item
  has a test in bugtests/<core|misc|tensor|reg|util|iterative>/<id>.cc (build.sh per directory).
- doc_questions.md: documentation questions with the author's answers.
- instructions, answers, answers2, answers3, optimize_discussion*.md: optimizer design discussion history.
- Bug fixing goes file by file. Done (2026-10-27): nonclass, vec, mat, smat, sparsemat, sparsesmat,
  tensor. Next: fourindex.h (C10, C12, F8), then permutation, polynomial/contfrac, small classes
  (vecmat3, quaternion, bitvector), regression (D1-D16), iterative (A26-A30), misc, remaining doc questions.

## User's application files (do not modify, only check that library changes do not break them)
- quantcomp.h (uses SparseSMat via gemv_n and the obsolete QCholesky), fcidumpcheck.cc.
