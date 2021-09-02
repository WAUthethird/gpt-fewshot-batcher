import trio
from datetime import datetime

import PySimpleGUI as sg


async def main() -> None:
    window = sg.Window(
        "Test",
        layout=[[sg.Button("Test", key="_TEST_")]],
        default_element_size=(100, 100),
        grab_anywhere=True,
        finalize=True,
    )
    while True:
        event, values = window.read(timeout=20)

        if event is None:
            break

        elif event == "_TEST_":
            print("Test")

        await trio.sleep(0.0)
    window.Close()


async def func() -> None:
    print(f"Enter {func=} function at {datetime.now()}.")
    await trio.sleep(10)
    print(f"Exit at {datetime.now()}")


async def parent() -> None:
    async with trio.open_nursery() as nursery:
        nursery.start_soon(main)
        nursery.start_soon(func)
        print(f"{parent=} here.")
    print(f"{parent=} exit.")

trio.run(parent)