using PyPlot

function explicitEulerAdaptiveStep(f, tspan, x0, h0, abstol, reltol, params)

    # Error controller parameter
    epstol = 0.8
    facmin = 0.1
    facmax = 5.0

    # Time interval
    t0 = tspan[1]
    tf = tspan[2]

    # Initial Conditions
    t = t0
    h = h0
    x = x0

    # Output
    T = [t]
    X = x

    while t < tf

        if t + h > tf
            h = tf - t
        end

        fEval = f(t,x,params)

        acceptStep = false
        while !acceptStep
            # Take step of size h
            x1 = x + h*fEval

            # Take step of size h/2
            hm = h / 2
            tm = t + hm
            xm = x + hm*fEval
            fm = f(tm, xm, params)
            x1hat = xm + hm * fm

            # Error estimation
            e = x1hat - x1
            r = max(abs(e)/max(abstol, abs(x1hat*reltol)))

            acceptStep = (r <= 1.0)
            if acceptStep
                t = t + h
                x = x1hat

                T = hcat(T, t)
                X = hcat(X, x)
            end

            # Asymptotic step size controller
            h = max(facmin, min(sqrt(epstol/r), facmax)) * h
        end
    end

    return (X, T)
end

function testFunction(t, x, params)
    return params[1]*x
end


# Call
tspan = [0,10]
x0 = 2
h0 = 1
abstol = 10e-3
reltol = 10e-3
params = [2]

(X,T) = explicitEulerAdaptiveStep(testFunction, tspan, x0, h0, abstol, reltol, params)
println(X[1])
println(typeof(T))
plot(X,color="red", linewidth=20.0, linestyle="--")
plot(X)
x = range(0; stop=2*pi, length=1000); y = sin.(3 * x + 4 * cos.(2 * x));
plot(x, y, color="red", linewidth=2.0, linestyle="--")
show()