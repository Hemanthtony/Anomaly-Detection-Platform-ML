import asyncio
import websockets
import json

async def test_websocket(anomaly_type):
    uri = f"ws://localhost:8003/ws/anomaly/{anomaly_type}"
    async with websockets.connect(uri) as websocket:
        if anomaly_type in ["network", "weather"]:
            # Send dummy features
            data = {"features": [0.5, 0.5, 0.5]}
        elif anomaly_type == "spam":
            # Send dummy text
            data = {"text": "This is a test spam message"}
        else:
            print(f"Unsupported anomaly type: {anomaly_type}")
            return

        await websocket.send(json.dumps(data))
        response = await websocket.recv()
        print(f"Response for {anomaly_type}: {response}")

async def main():
    await test_websocket("network")
    await test_websocket("weather")
    await test_websocket("spam")

if __name__ == "__main__":
    asyncio.run(main())
