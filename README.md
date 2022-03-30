KITT4SME Flaw Sleuth
--------------------
> leaves no stone unturned to find welding flaws!


### Hacking

Install Python (`>= 3.8`), Poetry (`>=1.1`) and the usual Docker
stack (Engine `>= 20.10`, Compose `>= 2.1`). If you've got Nix, you
get a dev shell with the right Python and Poetry versions simply by
running

```console
$ nix shell github:c0c0n3/kitt4sme.flaw-sleuth?dir=nix
```

Otherwise, install the usual way you do on your platform. Then clone
this repo, `cd` into its root dir and install the Python dependencies

```console
$ git clone https://github.com/c0c0n3/kitt4sme.flaw-sleuth.git
$ cd kitt4sme.flaw-sleuth
$ poetry install
```

Finally drop into a virtual env shell to hack away

```bash
$ poetry shell
$ charm .
# ^ Pycharm or whatever floats your boat
```

Run all the test suites:

```console
$ pytest tests
```

or just the unit tests

```console
$ pytest tests/unit
```

Measure global test coverage and generate an HTML report

```console
$ coverage run -m pytest -v tests
$ coverage html
```

Run the Streamlit app locally on port 8080 and without making it start
a browser

```console
$ streamlit run --server.headless true --server.port 8080 streamlit-app.py
```

Build and run the Docker image

```console
$ docker build -t kitt4sme/flaw-sleuth .
$ docker run -p 8501:8501 kitt4sme/flaw-sleuth
```
