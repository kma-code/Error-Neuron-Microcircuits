# error-mc


```
python3 -m venv MCenv
source MCenv/bin/activate
pip3 install -r requirements.txt
python -m ipykernel install --user --name=MCenv 
```
# todos
## KM:
- [x] make varphi' transfer optional
- [x] populations for E and I units (random lateral matrix)
- [x] fix error calculation in output (rate) and give layer-wise activation
- [x] enable non-hierarchical connections
- [x] make standalone runner
- [x] implement B = W.T for non-matching populations: $B = [1_{ER} \times W \times 1_{RE}]^T$
- [ ] noPAL: use same noise for every update
- [x] runner: implement test phase
- [ ] runner: implement compare for errormc
- [ ] implement skip forward connectivity
- [ ] realistic viz cx connectivity from Tang et al. 2024

## IJ:
- [x] debug vapi vs ubreveP - vbas (fix $\Delta w$)
  + we were missing the gapi conductance for scaling basal voltage [commit c2c22cc](https://github.com/kma-code/error-mc/commit/c2c22ccbaaabb263d4e03dc5802f07e0c6a680ab)
- [x] code cart-pole controller and use it to train a network
- [x] why we have weird spikes in vapi?
- [x] runner: cartpole
### Genn:
- [ ] errorMC neuron dynamics
- [ ] populations
- [ ] lateral weights: $L = 1 + \mathcal{N}(0, \sigma_{lat})$
- [ ] bw weights: FA and BP, where BP: $B = [1_{ER} \times W \times 1_{RE}]^T$
- [ ] varphi' transport
- [ ] $\Delta W = \varphi(u) - \varphi(v)$ or without $\varphi$


## General:

# Documentation

For learing rate `eta_fw`: errormc has `fw_connection_mode` `layered` and `skip`.
- For layered, `eta_fw` should be list with #entries same as `len(WPP_init)`.
- For skip, `eta_fw` should be list of lists with #entries same as `len(layers) x len(layers)`. The diagonal entries will be ignored (no recurrency), same for top-right (those would be backward connections).
