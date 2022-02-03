# v3d-web

Single camera motion capture/humanoid rigging in browser.

## Objectives

v3d-web is a project aiming to bridge the gap between our reality and the virtual world. With the power of latest XR and web technologies, you can render and rig your favorite 3D humanoid avatars right inside your browser. Try it out now!

## Features

- Contains a complete solution from image capturing to model rendering, all inside browsers.
- Thanks to the latest machine learning technologies, we can achieve a fairly accurate facial and pose estimation from a single camera video stream.
- It also comes with a complete WebGL rendering engine, in which VRM models can be present with highly complicated backgrounds. VRM model rendering is done with [v3d-core](https://github.com/phantom-software-AZ/v3d-core).
- There is a [demo site](https://www.phantom-dev.com) showing how this project can be seamlessly embedded into a modern UI framework like React.js.

## Usage

### Install from NPM

```s
npm install v3d-web
```

### In browser

*Only latest FireFox, Chrome/Chromium, Edge and Safari are supported. WebGL and WebAssembly are necessary for this project to work correctly.*

In the simplest case, all you need is:
```s
const vrmFile = 'testfile.vrm';
try {
    this.v3DWeb = new V3DWeb(vrmFile);
} catch (e: any) {
    console.error(e);
}
```

You will need HTML elements with certain `id`s. See [index.html](test/index.html).

A more complicated example can be found at the [repo for our demo site](https://github.com/phantom-software-AZ/v3d-web-demo).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Build

1. Clone this repo and submodules:

   ```s
   git clone https://github.com/phantom-software-AZ/v3d-web.git
   ```

2. Build v3d-web

   ```s
   npm install
   npm run build && tsc
   ```

## Debugging

Go to root folder of this repository and run:

```s
$ npm run debug
```

The debug page can be opened in any browser with `http://localhost:8080/`.

## Demo Site

See [this demo site](https://www.phantom-dev.com/demo) for a live example.

## Credits

- [MediaPipe](https://mediapipe.dev/)
- [babylon.js](https://github.com/BabylonJS/Babylon.js)
- [babylon-vrm-loader](https://github.com/virtual-cast/babylon-vrm-loader)
- [babylon-mtoon-material](https://github.com/virtual-cast/babylon-mtoon-material)
- [VRM Consortium](https://vrm.dev/en/)
- Demo model used: `Kaori/ Kaori` by `ClaValLuis` From VRoid Hub. Used according to [VRM PUBLIC LICENSE 1.0](https://vrm.dev/licenses/1.0/en/).

## Acknowledgement

An adorable and perseverant individual who keeps on pursuing dreams.

## Licenses

see [LICENSE](./LICENSE).
