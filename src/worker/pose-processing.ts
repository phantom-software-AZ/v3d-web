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

import * as Comlink from "comlink"
import {Results} from "@mediapipe/holistic";
import {Nullable} from "@babylonjs/core";

interface Poses {
    results: Nullable<Results>,
    process: () => void,
}

const poseResults : Poses = {
    results: null,
    process: function () {
        console.log(this.results);
    },
}




const obj = {
    counter: 0,
    inc(i: number = 1) {
        this.counter += i;
    },
    async spin() {
        let i = 0;
        while (i < 10000) {
            console.log(i);
            await sleep(i);
            i++;
        }
    }
};

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const exposeObj = {obj, poseResults};

Comlink.expose(exposeObj);

export {obj, poseResults, exposeObj};
