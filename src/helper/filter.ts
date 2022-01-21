/*
Copyright (C) 2022  The v3d Authors.

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

import {Vector3} from "@babylonjs/core";
import KalmanFilter from "kalmanjs";

export const VISIBILITY_THRESHOLD: number = 0.65;

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
 */
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
export class KalmanVectorFilter {
    private readonly kalmanFilterX;
    private readonly kalmanFilterY;
    private readonly kalmanFilterZ;
    constructor(
        public R = 0.1,
        public Q = 3,
    ) {
        this.kalmanFilterX = new KalmanFilter({Q: Q, R: R});
        this.kalmanFilterY = new KalmanFilter({Q: Q, R: R});
        this.kalmanFilterZ = new KalmanFilter({Q: Q, R: R});
    }

    public next(t: number, vec: Vector3) {
        const newValues = [
            this.kalmanFilterX.filter(vec.x),
            this.kalmanFilterY.filter(vec.y),
            this.kalmanFilterZ.filter(vec.z),
        ]

        return Vector3.FromArray(newValues);
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
        const len0 = ret.length();
        for (let i = 0; i < this.size; ++i) {
            ret.addInPlace(this.values[i].scale(this.kernel[i]));
        }
        const len1 = ret.length();
        // Normalize to original length
        ret.scaleInPlace(len0 / len1);

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
        if (this.value.subtract(v).length() > this.threshold) {
            this._value = v;
        }
    }

    public reset() {
        this._value = Vector3.Zero();
    }
}
