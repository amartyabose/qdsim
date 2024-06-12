import Pkg

ENV["PYTHON"] = ""
Pkg.add(["PyPlot", "PyCall"])

using qdsim

qdsim.comonicon_install()