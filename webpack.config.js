import * as path from 'path';
import { merge } from 'webpack-merge';
import { fileURLToPath } from 'url';
import {resolve} from "path";
import terser from "terser-webpack-plugin";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const baseConfig = {
    mode: 'production',
    entry: {
        v3dweb: path.resolve(__dirname, 'src', 'index'),
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: 'ts-loader',
            },
            {
                test: /\.m?js$/,
                resolve: {
                    fullySpecified: false,
                },
            },
        ],
    },
    resolve: {
        modules: [path.resolve(__dirname, 'node_modules')],
        extensions: ['.js', '.ts'],
        symlinks: false,
    },
    experiments: {
        topLevelAwait: true,
    },
    optimization: {
        minimize: true,
        minimizer: [new terser({
            extractComments: false,
        })],
        concatenateModules: true,
    },
    target: ['web'],
};

const config = [
    // UMD
    merge(baseConfig, {
        output: {
            library: {
                name: 'v3d-web',
                type: 'umd',
            },
            filename: '[name].module.js',
            path: path.resolve(__dirname, 'dist'),
        },
    }),
    // ES6
    merge(baseConfig, {
        output: {
            library: {
                type: 'module',
            },
            filename: '[name].es6.js',
            path: path.resolve(__dirname, 'dist'),
            environment: { module: true },
        },
        experiments: {
            outputModule: true,
        },
        externalsType: 'module',
    }),
    // browser global
    merge(baseConfig, {
        output: {
            library: {
                name: 'v3d-web',
                type: 'window',
            },
            filename: '[name].js',
            path: resolve(__dirname, 'dist'),
        },
    }),
];

export default config;
