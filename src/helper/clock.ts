/**
 * Adapted from three.js Clock class
 */
export class Clock {
    private _autoStart: boolean;
    get autoStart(): boolean {
        return this._autoStart;
    }
    private _startTime: number;
    get startTime(): number {
        return this._startTime;
    }
    private _oldTime: number;
    get oldTime(): number {
        return this._oldTime;
    }
    private _elapsedTime: number;
    get elapsedTime(): number {
        return this._elapsedTime;
    }
    private _running: boolean;
    get running(): boolean {
        return this._running;
    }

    constructor(autoStart = true) {
        this._autoStart = autoStart;

        this._startTime = 0;
        this._oldTime = 0;
        this._elapsedTime = 0;

        this._running = false;
    }

    public start() {
        this._startTime = now();

        this._oldTime = this._startTime;
        this._elapsedTime = 0;
        this._running = true;
    }

    public stop() {
        this.getElapsedTime();
        this._running = false;
        this._autoStart = false;
    }

    /**
     * Calculate how much time has passed from Clock start to input timestamp.
     * Unlike getElapsedTime, this method does not tick oldTime.
     * @param time input time
     */
    public getElapsedTimeAny(time: number) {
        return (time - this._startTime) / 1000;
    }

    public getElapsedTime() {
        this.getDelta();
        return this._elapsedTime;
    }

    public getDelta() {
        let diff = 0;

        if (this._autoStart && !this._running) {
            this.start();
            return 0;
        }

        if (this._running) {
            const newTime = now();
            diff = (newTime - this._oldTime) / 1000;
            this._oldTime = newTime;
            this._elapsedTime += diff;
        }

        return diff;
    }
}

export function now() {
    return (typeof performance === 'undefined' ? Date : performance).now(); // see #10732
}
