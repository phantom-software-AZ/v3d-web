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

import {Plane, Vector3, Curve3, ILoadingScreen} from "@babylonjs/core";

export function initArray<T>(length: number, initializer: (i: number) => T) {
    let arr = new Array<T>(length);
    for (let i = 0; i < length; i++)
        arr[i] = initializer(i);
    return arr;
}

export function range(start: number, end: number, step: number) {
    return Array.from(
        {length: Math.ceil((end - start) / step)},
        (_, i) => start + i * step
    );
}

export function linspace(start: number, end: number, div: number) {
    const step = (end - start) / div;
    return Array.from(
        {length: div},
        (_, i) => start + i * step
    );
}

export function objectFlip(obj: any) {
    const ret: any = {};
    Object.keys(obj).forEach((key: any) => {
        ret[obj[key]] = key;
    });
    return ret;
}

export const rangeCap = (
    v: number,
    min: number,
    max: number
) => {
    if (min > max) {
        const tmp = max;
        max = min;
        min = tmp;
    }
    return Math.max(Math.min(v, max), min);
}
export const remapRange = (
    v: number,
    src_low: number,
    src_high: number,
    dst_low: number,
    dst_high: number
) => {
    return dst_low + (v - src_low) * (dst_high - dst_low) / (src_high - src_low);
};
export const remapRangeWithCap = (
    v: number,
    src_low: number,
    src_high: number,
    dst_low: number,
    dst_high: number
) => {
    const v1 = rangeCap(v, src_low, src_high);
    return dst_low + (v1 - src_low) * (dst_high - dst_low) / (src_high - src_low);
};
export const remapRangeNoCap = (
    v: number,
    src_low: number,
    src_high: number,
    dst_low: number,
    dst_high: number
) => {
    return dst_low + (v - src_low) * (dst_high - dst_low) / (src_high - src_low);
};
export function validVector3(v: Vector3) {
    return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

export type KeysMatching<T, V> = { [K in keyof T]-?: T[K] extends V ? K : never }[keyof T];

// type MethodKeysOfA = KeysMatching<A, Function>;

export type IfEquals<X, Y, A = X, B = never> =
    (<T>() => T extends X ? 1 : 2) extends (<T>() => T extends Y ? 1 : 2) ? A : B;
export type ReadonlyKeys<T> = {
    [P in keyof T]-?: IfEquals<{ [Q in P]: T[P] }, { -readonly [Q in P]: T[P] }, never, P>
}[keyof T];

// type ReadonlyKeysOfA = ReadonlyKeys<A>;

export function setEqual<T>(as: Set<T>, bs: Set<T>) {
    if (as.size !== bs.size) return false;
    for (const a of as) if (!bs.has(a)) return false;
    return true;
}

export function projectVectorOnPlane(projPlane: Plane, vec: Vector3) {
    return vec.subtract(projPlane.normal.scale(Vector3.Dot(vec, projPlane.normal)));
}

export function round(value: number, precision: number) {
    const multiplier = Math.pow(10, precision || 0);
    return Math.round(value * multiplier) / multiplier;
}

/**
 * Simple fixed length FIFO queue.
 */
export class fixedLengthQueue<T> {
    private _values: T[] = [];
    get values(): T[] {
        return this._values;
    }

    constructor(public readonly size: number) {
    }

    public push(v: T) {
        this.values.push(v);

        if (this.values.length === this.size + 1) {
            this.values.shift();
        } else if (this.values.length > this.size + 1) {
            console.warn(`Internal queue has length longer than size ${this.size}: Got length ${this.values.length}`);
            this._values = this.values.slice(-this.size);
        }
    }

    public concat(arr: T[]) {
        this._values = this.values.concat(arr);

        if (this.values.length > this.size) {
            this._values = this.values.slice(-this.size);
        }
    }

    public pop() {
        return this.values.shift();
    }

    public first() {
        if (this._values.length > 0)
            return this.values[0];
        else
            return null;
    }

    public last() {
        if (this._values.length > 0)
            return this._values[this._values.length - 1];
        else
            return null;
    }

    public reset() {
        this.values.length = 0;
    }

    public length() {
        return this.values.length;
    }
}

export function findPoint(curve: Curve3, x: number, eps = 0.001) {
    const pts = curve.getPoints();
    if (x > pts[0].x) return pts[0].y;
    else if (x < pts[pts.length - 1].x) return pts[pts.length - 1].y;
    for (let i = 0; i < pts.length; ++i) {
        if (Math.abs(x - pts[i].x) < eps) return pts[i].y;
    }
    return 0;
}

export const LR = ["left", "right"];

// export class CustomLoadingScreen implements ILoadingScreen {
//     //optional, but needed due to interface definitions
//     public loadingUIBackgroundColor: string = '';
//     public loadingUIText: string = '';
//
//     private _loadingDiv = document.getElementById("loading");
//
//     constructor(private readonly renderingCanvas: HTMLCanvasElement) {}
//
//     public displayLoadingUI() {
//         if (!this._loadingDiv) return;
//         if (this._loadingDiv.style.display === 'none') {
//             // Do not add a loading screen if there is already one
//             this._loadingDiv.style.display = "initial";
//         }
//
//         this._resizeLoadingUI();
//         window.addEventListener("resize", this._resizeLoadingUI);
//     }
//
//     public hideLoadingUI() {
//         if (this._loadingDiv)
//             this._loadingDiv.style.display = "none";
//     }
//
//     private _resizeLoadingUI = () => {
//         const canvasRect = this.renderingCanvas.getBoundingClientRect();
//         const canvasPositioning = window.getComputedStyle(this.renderingCanvas).position;
//
//         if (!this._loadingDiv) {
//             return;
//         }
//
//         this._loadingDiv.style.position = (canvasPositioning === "fixed") ? "fixed" : "absolute";
//         this._loadingDiv.style.left = canvasRect.left + "px";
//         this._loadingDiv.style.top = canvasRect.top + "px";
//         this._loadingDiv.style.width = canvasRect.width + "px";
//         this._loadingDiv.style.height = canvasRect.height + "px";
//     }
// }

export function pointLineDistance(
    point: Vector3,
    lineEndA: Vector3, lineEndB: Vector3
) {
    const lineDir = lineEndB.subtract(lineEndA).normalize();
    const pProj = lineEndA.add(
        lineDir.scale(
            Vector3.Dot(point.subtract(lineEndA), lineDir)
            / Vector3.Dot(lineDir, lineDir)));
    return point.subtract(pProj).length();
}
