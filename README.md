# Elara Math

Elara Math is a Rust-native math library, with (current or planned support for):

- Tensors (n-dimensional arrays)\* (supported)
- Linear algebra with tensors (supported)
- Numerical solvers for integration & differential equations (WIP)
- Automatic differentiation (WIP)

\*: GPU tensors are not available yet, but GPU acceleration is planned to be added in the future

See the [neural network example](./examples/tiny_nn.rs) for a working tiny neural network using `elara-math`.

## Developing

To develop `elara-math`, first clone the repository:

```
git clone https://github.com/elaraproject/elara-gfx
```

Then, copy over the pre-commit githook:

```
cp .githooks/pre-commit .git/hooks/pre-commit && chmod a+x .git/hooks/pre-commit
```

You should then be all set to start making changes!
