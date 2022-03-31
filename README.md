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


### Live simulator

We've whipped together a test bed to simulate a live environment similar
to that of the KITT4SME cluster. In the `tests/sim` directory, you'll find
a Docker compose file with

* Orion LD connected to MongoDB
* Quantum Leap with a CrateDB backend
* Our Flaw Sleuth service

To start the show, run (Ctrl+C to stop)

```console
$ poetry shell
$ python tests/sim
```

This will bring up the Docker compose environment (assuming you've got a
Docker engine running already), subscribe Quantum Leap to Orion and then
will start sending welding machine entities to Orion. To see what's going
on, browse to the CrateDB Web UI at: http://localhost:4200.

Now browse to our Flaw Sleuth service page at: http://localhost:8501.
You should see a table containing the latest 10 entries in the welding
machine series Quantum Leap is busy collecting. Refresh the page to see
Flaw Sleuth refresh the table as Quantum Leap gets more data from Orion.