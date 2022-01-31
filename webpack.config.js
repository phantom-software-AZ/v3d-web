const path = require( 'path' );
const Merge = require('webpack-merge');
const terser = require('terser-webpack-plugin');

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
    Merge.merge(baseConfig, {
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
    Merge.merge(baseConfig, {
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
    Merge.merge(baseConfig, {
        output: {
            library: {
                name: 'v3d-web',
                type: 'window',
            },
            filename: '[name].js',
            path: path.resolve(__dirname, 'dist'),
        },
    }),
];

module.exports = config;
