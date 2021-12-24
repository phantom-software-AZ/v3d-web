/*
Copyright (C) 2021  The v3d Authors.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import {Mesh, MeshBuilder, Scene, StandardMaterial, Vector3} from "@babylonjs/core";
import {Color3, Vector4} from "@babylonjs/core/Maths";

type createSphereOptions = {
    segments?: number;
    diameter?: number;
    diameterX?: number;
    diameterY?: number;
    diameterZ?: number;
    arc?: number;
    slice?: number;
    sideOrientation?: number;
    frontUVs?: Vector4;
    backUVs?: Vector4;
    updatable?: boolean;
};

export function makeSphere(
    scene: Scene,
    pos?: Vector3,
    color?: number | string,
    options?: createSphereOptions) : Mesh {
    const sphere = MeshBuilder.CreateSphere("sphere",
        options || {
            diameterX: 1, diameterY: 0.5, diameterZ: 0.5
        }, scene);
    const material = new StandardMaterial("sphereMaterial", scene);
    if (color) {
        if (typeof color === 'number') color = `#${color.toString(16)}`;
        const thisColor = Color3.FromHexString(color);
        material.diffuseColor = thisColor;
    }
    sphere.material = material;

    if (pos)
        sphere.position = pos;

    return sphere;
}
