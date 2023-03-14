#ifndef Planar_SLAM_NLSSOLVER_IMPL_HPP_
#define Planar_SLAM_NLSSOLVER_IMPL_HPP_

// 对NLSSolver的实现


template<int D, typename T>
void NLLSSolver<D, T>::optimize(ModelType &model,float &_chi2) {
    if (method_ == GaussNewton) {
        optimizeGaussNewton(model,_chi2);//使用GaussNewton
    }
//    else if (method_ == LevenbergMarquardt) {
//        optimizeLevenbergMarquardt(model);
//    }
}

// Guass-Newton 优化
template<int D, typename T>
void NLLSSolver<D, T>::optimizeGaussNewton(ModelType &model,float &_chi2) {
    // Compute weight scale
    error_increased_ = false;
    //use_weights_ == false
    if (use_weights_)
        computeResiduals_zzw(model, false, true);


    // Save the old model to rollback in case of unsuccessful update
    ModelType old_model(model);
    // perform iterative estimation
    for (iter_ = 0; iter_ < n_iter_; ++iter_) {
        rho_ = 0;
        startIteration();

        //G-N优化过程中的H矩阵，和g矩阵（J*b）
        H_.setZero();
        Jres_.setZero();

        // compute initial error
        n_meas_ = 0;
        //根据两帧之间的初始位姿变换，计算两帧之间patch的残差并返回

        float new_chi2 = computeResiduals_zzw(model, true, false);
        _chi2 = new_chi2;

        // add prior
        //have_prior_ == false
        if (have_prior_)
            applyPrior(model);


        // solve the linear system
        if (!solve()) {
            //求解出现问题就退出迭代，具体就是出现nan
            // matrix was singular and could not be computed
            std::cout << "Matrix is close to singular! Stop Optimizing." << std::endl;
            std::cout << "H = " << H_ << std::endl;
            std::cout << "Jres = " << Jres_ << std::endl;
            std::cout << "x_ = " << x_ << std::endl;
            stop_ = true;
            getchar();
        }

        // check if error increased since last optimization
        // 判断cost是否增长了，正常来说光度误差是下降的，如上升则说明上一次迭代是最优的，并把上轮迭代结果输出
        if ((iter_ > 0 && new_chi2 > 1.2 * chi2_) || stop_) {
            if (verbose_) {
                std::cout << "It. " << iter_
                          << "\t Failure"
                          << "\t new_chi2 = " << new_chi2
                          << "\t Error increased. Stop optimizing."
                          << std::endl;
            }
            model = old_model; // rollback
            break;
        }

        // update the model
        // 如果光度误差下降了，则更新迭代结果
        ModelType new_model;
        update(model, new_model);
        //变量交换，准备下次迭代
        old_model = model;
        model = new_model;

        chi2_ = new_chi2;

//cout<<"\r"<<chi2_;
cout<<"iter_ "<<iter_<<"  chi2_ "<<chi2_<<"================================================="<<endl;
//if(iter_ +1 == n_iter_)
//    cout<<endl;
//getchar();

        verbose_ = false;
        if (verbose_) {
            std::cout << "It. " << iter_
                      << "\t Success"
                      << "\t new_chi2 = " << new_chi2
                      << "\t n_meas = " << n_meas_
                      << "\t x_norm = " << norm_max(x_)
                      << std::endl;
        }

        finishIteration();

        // stop when converged, i.e. update step too small
        if (norm_max(x_) <= eps_) {
            cout<<"update step too small"<<endl;
            break;
        }
    }
}

// LM 优化
/*
template<int D, typename T>
void NLLSSolver<D, T>::optimizeLevenbergMarquardt(ModelType &model) {
    // Compute weight scale
    error_increased_ = false;
    if (use_weights_) {
        computeResiduals(model, false, true);
    }

    // compute the initial error
    chi2_ = computeResiduals(model, true, false);

    if (verbose_)
        cout << "init chi2 = " << chi2_
             << "\t n_meas = " << n_meas_
             << endl;

    // TODO: compute initial lambda
    // Hartley and Zisserman: "A typical init value of lambda is 10^-3 times the
    // average of the diagonal elements of J'J"

    // Compute Initial Lambda
    if (mu_ < 0) {
        float H_max_diag = 0;
        float tau = 1e-4;
        for (size_t j = 0; j < D; ++j) {
            H_max_diag = max(H_max_diag, fabs(H_(j, j)));
        }
        mu_ = tau * H_max_diag;
    }

    // perform iterative estimation
    for (iter_ = 0; iter_ < n_iter_; ++iter_) {
        rho_ = 0;
        startIteration();

        // try to compute and update, if it fails, try with increased mu
        n_trials_ = 0;
        do {
            // init variables
            ModelType new_model;
            float new_chi2 = -1;
            H_.setZero();
            //H_ = mu_ * Matrix<double,D,D>::Identity(D,D);
            Jres_.setZero();

            // compute initial error
            n_meas_ = 0;
            computeResiduals(model, true, false);

            // add damping term:
            H_ += (H_.diagonal() * mu_).asDiagonal();

            // add prior
            if (have_prior_) {
                applyPrior(model);
            }

            // solve the linear system
            if (solve()) {
                // update the model
                update(model, new_model);

                // compute error with new model and compare to old error
                n_meas_ = 0;
                new_chi2 = computeResiduals(new_model, false, false);
                rho_ = chi2_ - new_chi2;
            } else {
                // matrix was singular and could not be computed
                cout << "Matrix is close to singular!" << endl;
                cout << "H = " << H_ << endl;
                cout << "Jres = " << Jres_ << endl;
                rho_ = -1;
            }

            if (rho_ > 0) {
                // update decrased the error -> success
                model = new_model;
                chi2_ = new_chi2;
                stop_ = norm_max(x_) <= eps_;
                mu_ *= max(1. / 3., min(1. - pow(2 * rho_ - 1, 3), 2. / 3.));
                nu_ = 2.;
                if (verbose_) {
                    cout << "It. " << iter_
                         << "\t Trial " << n_trials_
                         << "\t Success"
                         << "\t n_meas = " << n_meas_
                         << "\t new_chi2 = " << new_chi2
                         << "\t mu = " << mu_
                         << "\t nu = " << nu_
                         << endl;
                }
            } else {
                // update increased the error -> fail
                mu_ *= nu_;
                nu_ *= 2.;
                ++n_trials_;
                if (n_trials_ >= n_trials_max_) {
                    stop_ = true;
                }

                if (verbose_) {
                    cout << "It. " << iter_
                         << "\t Trial " << n_trials_
                         << "\t Failure"
                         << "\t n_meas = " << n_meas_
                         << "\t new_chi2 = " << new_chi2
                         << "\t mu = " << mu_
                         << "\t nu = " << nu_
                         << endl;
                }
            }

            finishTrial();

        } while (!(rho_ > 0 || stop_));
        if (stop_) {
            break;
        }

        finishIteration();
    }
}
*/

template<int D, typename T>
void NLLSSolver<D, T>::setRobustCostFunction(
        ScaleEstimatorType scale_estimator,
        WeightFunctionType weight_function) {
    switch (scale_estimator) {
        case TDistScale:
            if (verbose_) {
                printf("Using TDistribution Scale Estimator\n");
            }
            scale_estimator_.reset(new robust_cost::TDistributionScaleEstimator());
            use_weights_ = true;
            break;
        case MADScale:
            if (verbose_) {
                printf("Using MAD Scale Estimator\n");
            }
            scale_estimator_.reset(new robust_cost::MADScaleEstimator());
            use_weights_ = true;
            break;
        case NormalScale:
            if (verbose_) {
                printf("Using Normal Scale Estimator\n");
            }
            scale_estimator_.reset(new robust_cost::NormalDistributionScaleEstimator());
            use_weights_ = true;
            break;
        default:
            if (verbose_) {
                printf("Using Unit Scale Estimator\n");
            }
            scale_estimator_.reset(new robust_cost::UnitScaleEstimator());
            use_weights_ = false;
    }

    switch (weight_function) {
        case TDistWeight:
            if (verbose_) {
                printf("Using TDistribution Weight Function\n");
            }
            weight_function_.reset(new robust_cost::TDistributionWeightFunction());
            break;
        case TukeyWeight:
            if (verbose_) {
                printf("Using Tukey Weight Function\n");
            }
            weight_function_.reset(new robust_cost::TukeyWeightFunction());
            break;
        case HuberWeight:
            if (verbose_) {
                printf("Using Huber Weight Function\n");
            }
            weight_function_.reset(new robust_cost::HuberWeightFunction());
            break;
        default:
            if (verbose_) {
                printf("Using Unit Weight Function\n");
            }
            weight_function_.reset(new robust_cost::UnitWeightFunction());
    }
}

template<int D, typename T>
void NLLSSolver<D, T>::setPrior(
        const T &prior,
        const Matrix<float, D, D> &Information) {
    have_prior_ = true;
    prior_ = prior;
    I_prior_ = Information;
}

template<int D, typename T>
void NLLSSolver<D, T>::reset() {
    have_prior_ = false;
    chi2_ = 1e10;
    mu_ = mu_init_;
    nu_ = nu_init_;
    n_meas_ = 0;
    n_iter_ = n_iter_init_;
    iter_ = 0;
    stop_ = false;
}

template<int D, typename T>
inline const float &NLLSSolver<D, T>::getChi2() const {
    return chi2_;
}

template<int D, typename T>
inline const Matrix<float, D, D> &NLLSSolver<D, T>::getInformationMatrix() const {
    return H_;
}


#endif
