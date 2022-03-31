from fipy.docker import DockerCompose

from tests.util.fiware import wait_on_orion, wait_on_quantumleap, \
    WeldingMachineSampler, SubMan


docker = DockerCompose(__file__)


def bootstrap():
    docker.build_images()
    docker.start()

    wait_on_orion()
    wait_on_quantumleap()

    SubMan().create_subscriptions()


def send_welding_machine_entities():
    sampler = WeldingMachineSampler(pool_size=1)
    try:
        sampler.sample(samples_n=1000, sampling_rate=2.5)
    except Exception as e:
        print(e)


def run():
    services_running = False
    try:
        bootstrap()
        services_running = True

        print('>>> sending welding machine entities to Orion...')
        while True:
            send_welding_machine_entities()

    except KeyboardInterrupt:
        if services_running:
            docker.stop()
