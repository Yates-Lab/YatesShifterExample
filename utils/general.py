import torch
import numpy as np
import os

def ensure_tensor(x, device=None, dtype=None):
    """
    Ensures that the input is a torch.Tensor. If it is a numpy array, it is converted to a tensor.
    If device is provided, the tensor is moved to the device.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.
    device : torch.device
        The device to move the tensor to.
    dtype : torch.dtype
        The data type to convert the tensor to.

    Returns:
    -------
    torch.Tensor
        The input converted to a tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, int) or isinstance(x, float):
        x = torch.tensor([x])
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x

def ensure_ndarray(x, dtype=None):
    """
    Ensures that the input is a numpy.ndarray. If it is a tensor, it is converted to a numpy array.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.

    Returns:
    -------
    numpy.ndarray
        The input converted to a numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(x, int) or isinstance(x, float):
        x = [x]
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x
def set_seeds(seed:int):
    '''
    Set seeds for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def nd_cut(source_array, position, dest_shape, fill_value=0):
    """
    Extract a region of interest from an n-dimensional array. If the requested region extends
    beyond the source array bounds, the result is padded with fill_value.
    
    Args:
        source_array (np.ndarray): Source array to extract from
        position (tuple): Position in source array where to start extraction (coordinates of [0,0,...])
        dest_shape (tuple): Shape of the output array
        fill_value: Value to use for padding when outside source array bounds
        
    Returns:
        np.ndarray: Result array of shape dest_shape containing the extracted region
    """
    # Create output array filled with the fill value
    result = np.full(dest_shape, dtype=source_array.dtype, fill_value=fill_value)
    
    # Get number of dimensions
    ndim = len(dest_shape)
    
    # Convert inputs to numpy arrays for easier manipulation
    position = np.array(position)
    source_shape = np.array(source_array.shape)
    dest_shape = np.array(dest_shape)
    
    # Calculate effective source region
    source_start = np.maximum(np.zeros(ndim, dtype=int), position)
    source_end = np.minimum(source_shape, position + dest_shape)
    
    # Calculate effective target region
    target_start = np.maximum(np.zeros(ndim, dtype=int), -position)
    target_end = np.minimum(dest_shape, source_shape - position)
    
    # Create slicing tuples
    source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    
    # Perform the extraction operation only if there's a valid region to extract
    if all(end > start for start, end in zip(target_start, target_end)):
        result[target_slices] = source_array[source_slices]
    
    return result

def nd_paste(source_array, position, dest_shape, fill_value=0):
    """
    Paste an n-dimensional array into an array of specified shape at given position.
    
    Args:
        source_array (np.ndarray): Source array to paste
        position (tuple): Position where to paste the array (coordinates of [0,0,...])
        dest_shape (tuple): Shape of the output array
        
    Returns:
        np.ndarray: Result array of shape dest_shape with source_array pasted at position
    """
    # Create output array filled with zeros
    result = np.full(dest_shape, dtype=source_array.dtype, fill_value=fill_value)
    
    # Get number of dimensions
    ndim = len(dest_shape)
    
    # Convert inputs to numpy arrays for easier manipulation
    position = np.array(position)
    source_shape = np.array(source_array.shape)
    dest_shape = np.array(dest_shape)
    
    # Calculate effective source region (handling negative positions)
    source_start = np.maximum(np.zeros(ndim, dtype=int), -position)
    source_end = np.minimum(source_shape, dest_shape - position)
    
    # Calculate effective target region
    target_start = np.maximum(np.zeros(ndim, dtype=int), position)
    target_end = np.minimum(dest_shape, position + source_shape)
    
    # Create slicing tuples
    source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    
    # Perform the paste operation
    if all(end > start for start, end in zip(source_start, source_end)):
        result[target_slices] = source_array[source_slices]
    
    return result

def fit_sine(phases, spikes, omega=1.0, variance_source='observed_y'):
    """
    Fits a sine wave of the form f(x) = K*sin(omega*x) + L*cos(omega*x) + C
    (which can be rewritten as A*sin(omega*x + phi_0) + C) to the data
    using ordinary least squares (OLS). It then calculates the standard errors
    of the fitted parameters (K, L, C), derived quantities (Amplitude A,
    Phase offset phi_0), and a Modulation Index (MI).

    The method for calculating parameter variances depends on the 'variance_source'
    argument. For 'observed_y' or 'fitted_y', it uses a heteroscedasticity-
    consistent covariance estimator based on the idea that if beta_hat = M * y,
    then Cov(beta_hat) = M * Cov(y) * M^T, where Var(y_i) is approximated.
    For 'mse', it uses the standard OLS homoscedastic assumption.

    Parameters:
    -----------
    phases : array_like
        Observed phase values (x-values). These are typically in radians.
        If omega is not 1.0, these are the raw phase values, and the function
        will compute omega*phases.
    spikes : array_like
        Observed spike counts (y-values) corresponding to each phase.
    omega : float, optional
        Angular frequency of the sine wave (default is 1.0). If 'phases'
        already represent omega*x (i.e., phases are pre-multiplied by omega),
        then omega should be set to 1.0.
    variance_source : str, optional
        Method for estimating the variance of observations (Var(y_i)) when
        calculating the standard errors of the fitted parameters.
        - 'observed_y': Var(y_i) is approximated by y_i (the observed spike count).
                          This is suitable for Poisson-like data. Values are clipped
                          at a minimum of 1e-6 to prevent issues with zero counts. (Default)
        - 'fitted_y': Var(y_i) is approximated by the fitted y_i value. Clipped at 1e-6.
        - 'mse': Assumes homoscedastic errors (Var(y_i) = constant sigma_e^2).
                 sigma_e^2 is estimated by the Mean Squared Error (MSE) from OLS.
                 The covariance matrix of parameters is then MSE * (X^T X)^-1.

    Returns:
    --------
    dict
        A dictionary containing the fitted parameters, their standard errors,
        and other relevant statistics:
        - 'K': float, coefficient for the sin(omega*x) term.
        - 'L': float, coefficient for the cos(omega*x) term.
        - 'C': float, constant offset (baseline).
        - 'K_se': float, standard error of K.
        - 'L_se': float, standard error of L.
        - 'C_se': float, standard error of C.
        - 'amplitude': float, Amplitude A = sqrt(K^2 + L^2).
        - 'amplitude_se': float, Standard error of A, calculated via error propagation.
        - 'phase_offset_rad': float, Phase offset phi_0 = atan2(L, K) in radians.
                                The model can be written as A*sin(omega*x + phi_0) + C.
        - 'phase_offset_rad_se': float, Standard error of phi_0, via error propagation.
        - 'modulation_index': float, Modulation Index MI = 2A / (A+C).
                                NaN if A+C is close to zero, or if A or C is NaN.
        - 'modulation_index_se': float, Standard error of MI, via error propagation.
                                   NaN if MI is NaN or its variance cannot be computed.
        - 'y_fit': numpy.ndarray, The fitted spike counts (predicted y-values).
        - 'R_squared': float, Coefficient of determination (goodness of fit).
        - 'covariance_matrix_params': numpy.ndarray, The covariance matrix for [K, L, C].
        - 'condition_number_XTX': float, Condition number of the X^T*X matrix.
                                     High values (>1e8 to 1e10) may indicate
                                     multicollinearity and unstable parameter estimates.
    Raises:
    -------
    ValueError:
        If the number of data points is less than the number of parameters (3),
        if the X^T*X matrix is singular (e.g., phases are not distinct enough),
        or if 'variance_source' is an invalid option.
    """
    # Ensure inputs are numpy arrays for consistent operations
    phases = np.asarray(phases)
    spikes = np.asarray(spikes)
    n_points = len(phases)

    # Check if there are enough data points to fit the three parameters (K, L, C)
    if n_points < 3:
        raise ValueError("At least 3 data points are required to fit K, L, and C.")

    # Construct the design matrix X for the OLS regression.
    # The model is y = K*sin(omega*x) + L*cos(omega*x) + C*1
    # So, each row in X corresponds to an observation (phase, spike_count)
    # and columns are [sin(omega*phase_i), cos(omega*phase_i), 1]
    X = np.vstack([
        np.sin(omega * phases),  # First regressor: sin(omega*x)
        np.cos(omega * phases),  # Second regressor: cos(omega*x)
        np.ones(n_points)        # Third regressor: constant term (for C)
    ]).T  # Transpose to get observations in rows, regressors in columns

    # Calculate X^T * X (often denoted XTX)
    # This matrix is crucial for OLS parameter estimation.
    XTX = X.T @ X  # '@' is the matrix multiplication operator in Python 3.5+

    # Calculate the condition number of XTX.
    # The condition number gives an indication of the sensitivity of the solution
    # of a linear system to errors in the data. A high condition number
    # (e.g., > 1e8 or 1e10) suggests multicollinearity, meaning the regressors
    # are highly correlated, which can lead to unstable parameter estimates.
    condition_number_XTX = np.linalg.cond(XTX)
    if condition_number_XTX > 1e10: # A common threshold for concern
        print(f"Warning: Condition number of X^T*X is high ({condition_number_XTX:.2e}), "
              "results might be unstable. This can happen if phases are not well distributed "
              "or if omega is chosen such that sin(omega*x) and cos(omega*x) become "
              "linearly dependent for the given phases.")

    # Calculate the inverse of XTX, (X^T * X)^-1
    # This is needed for both parameter estimation and their covariance matrix.
    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        # This error occurs if XTX is singular (or nearly singular), meaning it cannot be inverted.
        # This typically happens if the columns of X are linearly dependent (perfect multicollinearity).
        # For example, if all phase values are the same or separated by multiples of 2pi/omega.
        raise ValueError("X^T*X matrix is singular. Cannot compute fit. "
                         "Check if phase values are sufficiently distinct and well-distributed.")

    # --- OLS Parameter Estimation ---
    # The OLS estimator for beta = [K, L, C]^T is beta_hat = (X^T*X)^-1 * X^T * y
    beta_hat = XTX_inv @ X.T @ spikes
    K, L, C = beta_hat[0], beta_hat[1], beta_hat[2]

    # Calculate the fitted y values (y_fit = X * beta_hat)
    y_fit = X @ beta_hat

    # --- Goodness of Fit: R-squared ---
    # R-squared = 1 - (Sum of Squared Residuals / Total Sum of Squares)
    # Sum of Squared Residuals (SSR) = sum((y_i - y_fit_i)^2)
    ss_residual = np.sum((spikes - y_fit)**2)
    # Total Sum of Squares (SST) = sum((y_i - mean(y))^2)
    ss_total = np.sum((spikes - np.mean(spikes))**2)
    
    if ss_total == 0:
        # Handle the case where all spike counts are the same.
        # If ss_total is 0, it means all y values are identical.
        # If ss_residual is also (close to) 0, the model fits perfectly (R^2=1).
        # Otherwise, the model does not explain any variance (R^2=0 can be ambiguous here,
        # but usually implies ss_residual > 0 if ss_total is 0, unless y_fit also all same).
        # A robust way is to say if residual is also zero, R2 is 1, else 0.
        r_squared = 1.0 if ss_residual < 1e-9 else 0.0 # Using a small tolerance for float comparison
    else:
        r_squared = 1.0 - (ss_residual / ss_total)

    # --- Covariance Matrix of Parameters [K, L, C] ---
    # This matrix provides variances of K, L, C on its diagonal,
    # and covariances (e.g., Cov(K,L)) on its off-diagonal.
    if variance_source == 'mse':
        # Homoscedastic assumption: Var(y_i) = sigma_e^2 (constant error variance)
        # Estimate sigma_e^2 using Mean Squared Error (MSE)
        # MSE = SSR / (degrees of freedom)
        # Degrees of freedom = n_points - number_of_parameters (which is 3: K, L, C)
        if n_points <= 3:
            # Not enough degrees of freedom to estimate MSE reliably.
            covariance_matrix_params = np.full((3,3), np.nan) # Fill with NaNs
            print("Warning: Cannot estimate MSE with <=3 data points for 3 parameters. "
                  "Parameter SEs will be NaN.")
        else:
            mse = ss_residual / (n_points - 3)
            # Cov(beta_hat) = MSE * (X^T*X)^-1
            covariance_matrix_params = mse * XTX_inv
    elif variance_source in ['observed_y', 'fitted_y']:
        # Heteroscedasticity-consistent covariance matrix (Eicker-Huber-White type)
        # Assumes errors can have non-constant variance.
        # Cov(beta_hat) = (X^T*X)^-1 * X^T * D * X * ((X^T*X)^-1)^T
        # where D is a diagonal matrix with Var(y_i) on the diagonal.
        # The formula can also be written as M * D * M^T where M = (X^T*X)^-1 * X^T.
        
        if variance_source == 'observed_y':
            # Approximate Var(y_i) = y_i (suitable for Poisson-distributed data)
            # Clip at a small positive value to avoid issues if y_i=0 (Var(0)=0 can be problematic).
            var_y_i = np.maximum(1e-6, spikes)
        else: # variance_source == 'fitted_y'
            # Approximate Var(y_i) = fitted_y_i
            var_y_i = np.maximum(1e-6, y_fit)
            
        D = np.diag(var_y_i) # Diagonal matrix of individual observation variances
        
        # M = (X^T*X)^-1 * X^T
        M = XTX_inv @ X.T
        # Cov(beta_hat) = M * D * M^T
        covariance_matrix_params = M @ D @ M.T
    else:
        raise ValueError("Invalid variance_source. Choose 'observed_y', 'fitted_y', or 'mse'.")

    # Extract variances of K, L, C from the diagonal of the covariance matrix
    var_K_val = covariance_matrix_params[0, 0]
    var_L_val = covariance_matrix_params[1, 1]
    var_C_val = covariance_matrix_params[2, 2]
    
    # Standard errors are the square roots of these variances.
    # Ensure variance is non-negative before taking sqrt; otherwise, SE is NaN.
    K_se = np.sqrt(var_K_val) if var_K_val >= 0 else np.nan
    L_se = np.sqrt(var_L_val) if var_L_val >= 0 else np.nan
    C_se = np.sqrt(var_C_val) if var_C_val >= 0 else np.nan

    # --- Amplitude (A) and its Standard Error ---
    # A = sqrt(K^2 + L^2)
    amplitude = np.sqrt(K**2 + L**2)
    amplitude_se = np.nan # Initialize to NaN
    var_A = np.nan        # Initialize variance of A to NaN

    # For error propagation, we need derivatives of A with respect to K and L.
    # dA/dK = K / sqrt(K^2+L^2) = K / A
    # dA/dL = L / sqrt(K^2+L^2) = L / A
    # To avoid division by zero if A is very small, use a clipped amplitude_denom.
    amplitude_denom = max(amplitude, 1e-9) 

    dAdK = K / amplitude_denom
    dAdL = L / amplitude_denom
    
    # Check if component variances/covariance are NaN (e.g., if MSE calculation failed)
    if np.isnan(var_K_val) or np.isnan(var_L_val) or np.isnan(covariance_matrix_params[0,1]):
        # var_A will remain NaN, and thus amplitude_se will remain NaN
        pass
    else:
        cov_KL = covariance_matrix_params[0, 1] # Covariance between K and L
        # Var(A) approx (dA/dK)^2*Var(K) + (dA/dL)^2*Var(L) + 2*(dA/dK)*(dA/dL)*Cov(K,L)
        var_A = (dAdK**2 * var_K_val) + \
                (dAdL**2 * var_L_val) + \
                (2 * dAdK * dAdL * cov_KL)
        if var_A >= 0:
            amplitude_se = np.sqrt(var_A)
        # If var_A computed is negative (due to numerical issues or model misspecification),
        # amplitude_se remains NaN.

    # Warning if amplitude is very small, as its SE might be unreliable.
    if amplitude < 1e-9 and not np.isnan(amplitude): # Check if amplitude itself isn't already NaN
         print("Warning: Amplitude is close to zero. Standard error for amplitude and phase may be unreliable or NaN.")

    # --- Phase Offset (phi_0) and its Standard Error ---
    # The model can be written as A*sin(omega*x + phi_0) + C.
    # K = A*cos(phi_0), L = A*sin(phi_0) => phi_0 = atan2(L, K)
    # atan2 is used for numerical stability and correct quadrant.
    phase_offset_rad = np.arctan2(L, K)
    phase_offset_rad_se = np.nan # Initialize to NaN
    
    # For error propagation, derivatives of phi_0 w.r.t. K and L:
    # d(phi_0)/dK = -L / (K^2+L^2) = -L / A^2
    # d(phi_0)/dL =  K / (K^2+L^2) =  K / A^2
    # Use amplitude_denom^2 for A^2 to avoid division by zero.
    amplitude_sq_denom = amplitude_denom**2

    dphidK = -L / amplitude_sq_denom
    dphidL = K / amplitude_sq_denom

    if np.isnan(var_K_val) or np.isnan(var_L_val) or np.isnan(covariance_matrix_params[0,1]):
        # phase_offset_rad_se will remain NaN
        pass
    else:
        cov_KL = covariance_matrix_params[0, 1] # Cov(K,L)
        # Var(phi_0) approx (dphi/dK)^2*Var(K) + (dphi/dL)^2*Var(L) + 2*(dphi/dK)*(dphi/dL)*Cov(K,L)
        var_phi0 = (dphidK**2 * var_K_val) + \
                   (dphidL**2 * var_L_val) + \
                   (2 * dphidK * dphidL * cov_KL)
        if var_phi0 >= 0:
            phase_offset_rad_se = np.sqrt(var_phi0)
        # If var_phi0 is negative, phase_offset_rad_se remains NaN.


    # --- Modulation Index (MI) and its Standard Error ---
    # MI = 2*A / (A+C)
    modulation_index = np.nan
    modulation_index_se = np.nan
    cov_AC = np.nan # Covariance between Amplitude (A) and Offset (C)

    # Calculate Cov(A,C) using error propagation:
    # Cov(A,C) approx (dA/dK)*Cov(K,C) + (dA/dL)*Cov(L,C)
    # dAdK and dAdL were computed earlier for amplitude_se.
    if not (np.isnan(K) or np.isnan(L) or \
            np.isnan(covariance_matrix_params[0,2]) or np.isnan(covariance_matrix_params[1,2]) or \
            np.isnan(dAdK) or np.isnan(dAdL) ): # dAdK/L can be NaN if K/L are NaN or amplitude is NaN initially
        cov_KC = covariance_matrix_params[0, 2]  # Cov(K, C)
        cov_LC = covariance_matrix_params[1, 2]  # Cov(L, C)
        cov_AC = dAdK * cov_KC + dAdL * cov_LC
    
    # Proceed if amplitude and C are valid numbers
    if np.isnan(amplitude) or np.isnan(C):
        # MI and MI_se remain NaN if A or C is NaN (e.g. due to upstream NaN K,L)
        pass # This state implies that K,L,C or their SEs might already be NaN
    else:
        # Denominator for MI: A+C
        denom_MI_val = amplitude + C
        
        # Check if A+C is too small (close to zero).
        # Since A (amplitude) >= 0 and C (offset, typically mean firing rate) >= 0,
        # A+C is usually non-negative.
        if denom_MI_val < 1e-9: 
            print(f"Warning: Sum of amplitude ({amplitude:.2e}) and C ({C:.2e}) is close to zero. "
                  "Modulation index and its SE are set to NaN.")
            # modulation_index and modulation_index_se remain np.nan
        else:
            # Calculate Modulation Index
            modulation_index = (2 * amplitude) / denom_MI_val
            
            # For standard error of MI, calculate partial derivatives:
            # d(MI)/dA = 2*C / (A+C)^2
            # d(MI)/dC = -2*A / (A+C)^2
            denom_MI_sq = denom_MI_val**2 # (A+C)^2
            dMI_dA = (2 * C) / denom_MI_sq
            dMI_dC = (-2 * amplitude) / denom_MI_sq

            # Check if required variances (Var(A), Var(C)) and Cov(A,C) are available (not NaN)
            # var_A was computed for amplitude_se
            # var_C_val is covariance_matrix_params[2,2] (variance of C)
            if np.isnan(var_A) or np.isnan(var_C_val) or np.isnan(cov_AC):
                # modulation_index_se remains NaN
                pass
            else:
                # Var(MI) approx (dMI/dA)^2*Var(A) + (dMI/dC)^2*Var(C) + 2*(dMI/dA)*(dMI/dC)*Cov(A,C)
                var_modulation_index = (dMI_dA**2 * var_A) + \
                                   (dMI_dC**2 * var_C_val) + \
                                   (2 * dMI_dA * dMI_dC * cov_AC)
                
                if var_modulation_index >= 0:
                    modulation_index_se = np.sqrt(var_modulation_index)
                else:
                    # This can happen due to numerical instability or if the model is ill-conditioned.
                    if not np.isnan(var_modulation_index): # Only print if it's a negative number, not already NaN
                        print(f"Warning: Calculated variance for modulation index is negative ({var_modulation_index:.2e}). "
                              "Setting SE to NaN. This may indicate issues with model stability or covariance estimates.")
                    # modulation_index_se remains NaN (or its initial np.nan state)
                    
    # --- Return Results ---
    return {
        'K': K, 'L': L, 'C': C,
        'K_se': K_se, 'L_se': L_se, 'C_se': C_se,
        'amplitude': amplitude, 'amplitude_se': amplitude_se,
        'phase_offset_rad': phase_offset_rad, 'phase_offset_rad_se': phase_offset_rad_se,
        'modulation_index': modulation_index, 'modulation_index_se': modulation_index_se,
        'y_fit': y_fit, 'R_squared': r_squared,
        'covariance_matrix_params': covariance_matrix_params,
        'condition_number_XTX': condition_number_XTX
    }

