# Todo

## Frontend
- [ ] Implement lexer
- [ ] Implement parser
    + [ ] Standardize syntax of `llfpl` source files
- [ ] Refine AST so as to make it easier: maybe go from CST -> AST given that encoding so far relies on the AST?

## Backend
- [ ] Implement encoding
    + [ ] Levels are encoded using Fractional Binding right now, handroll RHC for use with torchhd
- [ ] Implement type checking
- [ ] Implement interpretation
- [ ] Implement decoding

## Testing

- [ ] Generate dataset of sample source programs
- [ ] Create model to infer the type of programs
- [ ] Investigate features of the model