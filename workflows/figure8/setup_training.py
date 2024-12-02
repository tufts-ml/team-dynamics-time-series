from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings

n_cavi_iterations = 10
num_M_step_iters = 50

M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_CSP = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"

Mstep_toggles = M_step_toggles_from_strings(
    M_step_toggle_for_STP,
    M_step_toggle_for_ETP,
    M_step_toggle_for_CSP,
    M_step_toggle_for_IP,
	)