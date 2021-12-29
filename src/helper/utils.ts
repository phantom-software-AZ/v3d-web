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

import {Angle, Mesh, MeshBuilder, Nullable, PrecisionDate, Quaternion, Scene, Vector3} from "@babylonjs/core";
import {NormalizedLandmark} from "@mediapipe/holistic";
import {CloneableQuaternion} from "../worker/pose-processing";

export function initArray<T>(length: number, initializer: (i: number) => T) {
    let arr = new Array<T>(length);
    for (let i = 0; i < length; i++)
        arr[i] = initializer(i);
    return arr;
}


export const gaussianKernel1d = (function () {
    let sqr2pi = Math.sqrt(2 * Math.PI);

    return function gaussianKernel1d (size: number, sigma: number) {
        // ensure size is even and prepare variables
        let width = (size / 2) | 0,
            kernel = new Array(width * 2 + 1),
            norm = 1.0 / (sqr2pi * sigma),
            coefficient = 2 * sigma * sigma,
            total = 0,
            x;

        // set values and increment total
        for (x = -width; x <= width; x++) {
            total += kernel[width + x] = norm * Math.exp(-x * x / coefficient);
        }

        // divide by total to make sure the sum of all the values is equal to 1
        for (x = 0; x < kernel.length; x++) {
            kernel[x] /= total;
        }

        return kernel;
    };
}());

/*
 * Converted from https://github.com/jaantollander/OneEuroFilter.
 * Licensed under MIT.
 */
export class OneEuroFilter {
    constructor(
        public t_prev: number,
        public x_prev: number,
        private dx_prev = 0.0,
        private readonly min_cutoff = 1.0,
        private readonly beta = 0.0,
        private readonly d_cutoff = 1.0
    ) {
    }

    private static smoothing_factor(t_e: number, cutoff: number) {
        const r = 2 * Math.PI * cutoff * t_e;
        return r / (r + 1);
    }

    private static exponential_smoothing(a: number, x: number, x_prev: number) {
        return a * x + (1 - a) * x_prev;
    }

    public next(t: number, x: number) {
        const t_e = t - this.t_prev;

        // The filtered derivative of the signal.
        const a_d = OneEuroFilter.smoothing_factor(t_e, this.d_cutoff);
        const dx = (x - this.x_prev) / t_e;
        const dx_hat = OneEuroFilter.exponential_smoothing(a_d, dx, this.dx_prev);

        // The filtered signal.
        const cutoff = this.min_cutoff + this.beta * Math.abs(dx_hat);
        const a = OneEuroFilter.smoothing_factor(t_e, cutoff);
        const x_hat = OneEuroFilter.exponential_smoothing(a, x, this.x_prev);

        // Memorize the previous values.
        this.x_prev = x_hat;
        this.dx_prev = dx_hat;
        this.t_prev = t;

        return x_hat;
    }
}

export class OneEuroVectorFilter {
    constructor(
        public t_prev: number,
        public x_prev: Vector3,
        private dx_prev = Vector3.Zero(),
        public min_cutoff = 1.0,
        public beta = 0.0,
        public d_cutoff = 1.0
    ) {
    }

    private static smoothing_factor(t_e: number, cutoff: number) {
        const r = 2 * Math.PI * cutoff * t_e;
        return r / (r + 1);
    }

    private static exponential_smoothing(a: number, x: Vector3, x_prev: Vector3) {
        return x.scale(a).addInPlace(x_prev.scale((1 - a)));
    }

    public next(t: number, x: Vector3) {
        const t_e = t - this.t_prev;

        // The filtered derivative of the signal.
        const a_d = OneEuroVectorFilter.smoothing_factor(t_e, this.d_cutoff);
        const dx = x.subtract(this.x_prev).scaleInPlace(1 / t_e);
        const dx_hat = OneEuroVectorFilter.exponential_smoothing(a_d, dx, this.dx_prev);

        // The filtered signal.
        const cutoff = this.min_cutoff + this.beta * dx_hat.length();
        const a = OneEuroVectorFilter.smoothing_factor(t_e, cutoff);
        const x_hat = OneEuroVectorFilter.exponential_smoothing(a, x, this.x_prev);

        // Memorize the previous values.
        this.x_prev = x_hat;
        this.dx_prev = dx_hat;
        this.t_prev = t;

        return x_hat;
    }
}

export class GaussianVectorFilter {
    private _values: Vector3[] = [];
    get values(): Vector3[] {
        return this._values;
    }
    private readonly kernel: number[];

    constructor(
        public readonly size: number,
        private readonly sigma: number
    ) {
        if (size < 2) throw RangeError("Filter size too short");
        this.size = Math.floor(size);
        this.kernel = gaussianKernel1d(size, sigma);
    }

    public push(v: Vector3) {
        this.values.push(v);

        if (this.values.length === this.size + 1) {
            this.values.shift();
        } else if (this.values.length > this.size + 1) {
            console.warn(`Internal queue has length longer than size: ${this.size}`);
            this.values.slice(-this.size);
        }
    }

    public reset() {
        this.values.length = 0;
    }

    public apply() {
        if (this.values.length !== this.size) return Vector3.Zero();
        const ret = this.values[0].clone();
        for (let i = 0; i < this.size; ++i) {
            ret.addInPlace(this.values[i].scale(this.kernel[i]));
        }

        return ret;
    }
}

export class EuclideanHighPassFilter {
    private _value: Vector3 = Vector3.Zero();
    get value(): Vector3 {
        return this._value;
    }

    constructor(
        private readonly threshold: number
    ) {}

    public update(v: Vector3) {
        // console.log(this.value.subtract(v).length());
        if (this.value.subtract(v).length() > this.threshold) {
            this._value = v;
        }
    }

    public reset() {
        this._value = Vector3.Zero();
    }
}

export class FrameMonitor {
    private _lastFrameTimeMs: Nullable<number> = null;

    // Shall only be called once per onResults()
    public sampleFrame(timeMs: number = PrecisionDate.Now) {
        let dt = 0;
        if (this._lastFrameTimeMs !== null) {
            dt = timeMs - this._lastFrameTimeMs;
        }

        this._lastFrameTimeMs = timeMs;
        return dt;
    }
}

export const POSE_LANDMARK_LENGTH = 33;
export const FACE_LANDMARK_LENGTH = 478;
export const HAND_LANDMARK_LENGTH = 21;

export class Arrow3D {
    //Shape profile in XY plane
    private readonly myShape: Vector3[] = [];
    private readonly myPath: Vector3[] = [];
    private arrowInstance: Nullable<Mesh> = null;

    private _arrowRadius: number;
    get arrowRadius(): number {
        return this._arrowRadius;
    }
    set arrowRadius(value: number) {
        this._arrowRadius = value;
        this.updateTopShape();
    }
    private _n: number;
    get n(): number {
        return this._n;
    }
    set n(value: number) {
        this._n = value;
        this.updateTopShape();
    }

    private _arrowHeadLength: number;
    get arrowDirection(): Vector3 {
        return this._arrowDirection;
    }
    set arrowDirection(value: Vector3) {
        this._arrowDirection = value;
        this.updatePath();
    }
    private _arrowHeadMaxSize: number;
    get arrowStart(): Vector3 {
        return this._arrowStart;
    }
    set arrowStart(value: Vector3) {
        this._arrowStart = value;
        this.updatePath();
    }
    private _arrowLength: number;
    get arrowLength(): number {
        return this._arrowLength;
    }
    set arrowLength(value: number) {
        this._arrowLength = value;
        this.updatePath();
    }
    private _arrowStart: Vector3;
    get arrowHeadMaxSize(): number {
        return this._arrowHeadMaxSize;
    }
    set arrowHeadMaxSize(value: number) {
        this._arrowHeadMaxSize = value;
        this.updatePath();
    }
    private _arrowDirection: Vector3;
    get arrowHeadLength(): number {
        return this._arrowHeadLength;
    }
    set arrowHeadLength(value: number) {
        this._arrowHeadLength = value;
        this.updatePath();
    }

    constructor(
        private scene: Scene,
        arrowRadius = 0.5,
        n = 30,
        arrowHeadLength = 1.5,
        arrowHeadMaxSize = 1.5,
        arrowLength = 10,
        arrowStart: Vector3,
        arrowDirection: Vector3,
    ) {
        this._arrowRadius = arrowRadius;
        this._n = n;
        this._arrowHeadLength = arrowHeadLength;
        this._arrowHeadMaxSize = arrowHeadMaxSize;
        this._arrowLength = arrowLength;
        this._arrowStart = arrowStart;
        this._arrowDirection = arrowDirection;
        this.updateTopShape();
        this.updatePath();
    }

    private updateTopShape() {
        const deltaAngle = 2 * Math.PI / this.n;
        this.myShape.length = 0;
        for (let i = 0; i <= this.n; i++) {
            this.myShape.push(new Vector3(
                this.arrowRadius * Math.cos(i * deltaAngle),
                this.arrowRadius * Math.sin(i * deltaAngle),
                0))
        }
        this.myShape.push(this.myShape[0]);  //close profile
    }

    private updatePath() {
        const arrowBodyLength = this.arrowLength - this.arrowHeadLength;
        this.arrowDirection.normalize();
        const arrowBodyEnd = this.arrowStart.add(this.arrowDirection.scale(arrowBodyLength));
        const arrowHeadEnd = arrowBodyEnd.add(this.arrowDirection.scale(this.arrowHeadLength));

        this.myPath.length = 0;
        this.myPath.push(this.arrowStart);
        this.myPath.push(arrowBodyEnd);
        this.myPath.push(arrowBodyEnd);
        this.myPath.push(arrowHeadEnd);

        if (!this.arrowInstance)
            this.arrowInstance = MeshBuilder.ExtrudeShapeCustom(
                "arrow",
                {
                    shape: this.myShape,
                    path: this.myPath,
                    updatable: true,
                    scaleFunction: this.scaling.bind(this),
                    sideOrientation: Mesh.DOUBLESIDE}, this.scene);
        else
            this.arrowInstance = MeshBuilder.ExtrudeShapeCustom(
                "arrow",
                {
                    shape: this.myShape,
                    path: this.myPath,
                    scaleFunction: this.scaling.bind(this),
                    instance: this.arrowInstance}, this.scene);
    }

    private scaling(index: number, distance: number): number {
        switch (index) {
            case 0:
            case 1:
                return 1;
            case 2:
                return this.arrowHeadMaxSize / this.arrowRadius;
            case 3:
                return 0;
            default:
                return 1;
        }
    }

    public updateStartAndDirection(arrowStart: Vector3, arrowDirection: Vector3) {
        this._arrowStart = arrowStart;
        this._arrowDirection = arrowDirection;
        this.updatePath();
    }
}

export const normalizedLandmarkToVector = (
    l: NormalizedLandmark,
    scaling = 1.,
    reverseY = false) => {
    return new Vector3(
        l.x * scaling,
        reverseY ? -l.y * scaling : l.y * scaling,
        l.z * scaling);
}
export const vectorToNormalizedLandmark = (l: Vector3) : NormalizedLandmark => {
    return {x: l.x, y: l.y, z: l.z};
};
export const cloneableQuaternionToQuaternion = (q: CloneableQuaternion): Quaternion => {
    return new Quaternion(q.x, q.y, q.z, q.w);
};

export function range(start: number, end: number, step: number) {
    return Array.from(
        { length: Math.ceil((end - start) / step) },
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

export const HAND_LANDMARKS = {
    WRIST: 0,
    THUMB_CMC: 1,
    THUMB_MCP: 2,
    THUMB_IP: 3,
    THUMB_TIP: 4,
    INDEX_FINGER_MCP: 5,
    INDEX_FINGER_PIP: 6,
    INDEX_FINGER_DIP: 7,
    INDEX_FINGER_TIP: 8,
    MIDDLE_FINGER_MCP: 9,
    MIDDLE_FINGER_PIP: 10,
    MIDDLE_FINGER_DIP: 11,
    MIDDLE_FINGER_TIP: 12,
    RING_FINGER_MCP: 13,
    RING_FINGER_PIP: 14,
    RING_FINGER_DIP: 15,
    RING_FINGER_TIP: 16,
    PINKY_MCP: 17,
    PINKY_PIP: 18,
    PINKY_DIP: 19,
    PINKY_TIP: 20,
};

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

// Same as three.js Quaternion.setFromUnitVectors
export const quaternionBetweenVectors = (
    v1: Vector3, v2: Vector3
): Quaternion => {
    const angle = Math.acos(Vector3.Dot(v1, v2));
    const axis = Vector3.Cross(v1,v2);
    return Quaternion.RotationAxis(axis, angle);
};
// From -180 to 180
export const remapDegreeWithCap = (deg: number) => {
    deg = rangeCap(deg, 0, 360);
    return deg < 180 ? deg : deg - 360;
}
export const degreeBetweenVectors = (
    v1: Vector3, v2: Vector3
) => {
    const angles = quaternionBetweenVectors(v1, v2).toEulerAngles();
    return new Vector3(
        remapDegreeWithCap(Angle.FromRadians(angles.x).degrees()),
        remapDegreeWithCap(Angle.FromRadians(angles.y).degrees()),
        remapDegreeWithCap(Angle.FromRadians(angles.z).degrees()),
    );
};
