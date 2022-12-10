# ezDiff

A tiny forward automatic differentiation library for learning purposes. If you need something fully featured, check out [`hyperdual`](https://crates.io/crates/hyperdual).

## What is automatic differentiation?

AutoDiff is a way to automatically calculate derivatives. AutoDiff is *not* symbolic (computer algebra) or numerical differentiation, instead it computes derivatives at the same time as the regular values are being evaluated, by using dual numbers.

### Dual Numbers

They sound fancy, but it's essentially just using a tuple of values `(x, dx)`, instead of a single value. Instead of computing `f(x) = y` and passing in `x` to find `y`, we pass in a tuple `(x, dx)` and compute  *both* `(y, dy/dx)` at the same time using operator overloading.

### Operator Overloading

Operator overloading is a technique where you *overload* mathematical operators in a programming language (`+`, `-`, `*`, `/`) with your own implementation. To do this in Rust, all we need to do is take our dual number type, and implement the math operation traits on it: `impl Add for Dual { ... }`.

### How does it actually *work*?

So, we've talked in abstract how we replace `x` when evaluating `f(x)` with a tuple `(x, dx)`, then do *something* to that number with operator overloading. This is where we get to the neat trick (mathematicians *hate* this one weird trick). The trick is conceptually simple - use the chain rule.

The chain rule tells us that when we are trying to find the derivative of some really complicated function, we can split up the complicated function into many small, simple functions that are easy to evaluate. 

$\frac{d}{dx}[f(g(x))] = f'(g(x)) * g'(x)$

Let's say we have some function $y = cos(x^2)$. I don't know how to find the derivative of that, so let's split it up into some smaller functions I do know how to evaluate. Let's say:

$f(g(x)) = cos(g(x))$ and $g(x) = x^2$

We no longer need to find the derivative of $cos(x^2)$! Now we only need to find the derivative of $cos(u)$ and $x^2$ separately, then mush them together.

So, how do dual numbers come into play here? If you take a look at how we evaluated $cos(x^2)$ by breaking it into smaller parts, well, that's also how we evaluate a function normally! I can't easily compute $cos(x^2)$, but I can compute $x^2$, then plug it into $cos()$. We can take advantage of this to compute the "primal" (x) and the derivative (d/dx). Let's do this step-by-step:

1. Let's start by defining a dual number as `Dual = (x, dx)`.
2. Plug it into the function we want to evaluate: `cos(Dual^2)`
3. We can begin evaluating by following the normal order of operations, and find the value of `Dual^2`
    
    `Dual^2 = (x^2, 2*x*dx)`
    
    Here we calculate the primal value on the left, and the derivative on the right. The primal is just, well, the normal operation, `x^2`. For the derivative, all we need to do is answer: what is the *derivative* of `x^2`? Here we can simply use the power rule: $\frac{d}{dx} x^n = nx^{n-1} dx$. Concretely, $\frac{d}{dx} x^2 = 2*x* dx$.

4. We can continue with the order of operations and evaluate $cos(Dual)$, using the rules for derivatives of trig functions: $\frac{d}{dx}cos(x) = -sin(x)dx$.

    `Dual.cos() = (x.cos(), -x.sin() * dx)`

    Now, remember at this point `Dual` already has values inside it from when we evaluated `Dual^2`. When we substitute that in, we see that our `Dual` contains:

    `Dual.x = (x^2).cos()`

    `Dual.dx = -x.sin() * (2*x*dx)`

    Anywhere we see `x` or `dx`, we replace that with the previous value stored in the dual number.

5. That's really all there is to it. If we plug in our initial conditions into `Dual`, let's say at $x = 5$ we start with `Dual(5, 1)`. Note the derivative always starts as $1$.

    `y = (x^2).cos() = 0.9912028118634736`

    `dy/dx = -x.sin() * (2*x*dx) = 1.3235175009777302`

In this example, we only needed to know the derivative of $cos(x)$ and $x^2$. We encode this by overloading those operators (defining math operators for our `Dual` type), and letting the language evaluate our function normally. All we have to do is:

```rs
let x = Dual::new(5.0);
let y = (x.pow(2.0)).cos();
dbg!(y);
```

which returns

```
y = Dual {
    val: 0.9912028118634736,
    dot: 1.3235175009777302,
}
```

## Why is this useful?

Instead of needing to iteratively approximate the value (numerical, finite differences), or attempt to find a symbolic representation of the derivative (computer algebra), we can compute the derivative damn near for free, and the compiler can optimize it inline with our code. Because the implementation is just defining the derivative as a math operation the same way the equation would normally be evaluated, the computational complexity of the derivative is proportional to the complexity of the original equation!