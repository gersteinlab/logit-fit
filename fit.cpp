/* A data structure for holding the return values of the glm.fit function */
class fit {
	private: // Member variables
		vector<double> coefficients;
		vector<double> residuals;
		vector<double> fitted_values;
		vector<double> effects;
		vector<vector<double> > R;
		int rank;
		vector<vector<double> > qr;
		vector<double> qraux;
		vector<int> pivot;
		double tol;
		vector<double> linear_predictors;
		double deviance;
		double aic;
		double null_deviance;
		int iter;
		vector<double> weights;
		vector<double> prior_weights;
		int df_residual;
		int df_null;
		vector<double> y;
		bool converged;
		bool boundary;
		double theta;
		double se_theta;
		double twologlik;
		
		public: // Set/get methods
			vector<double> getCoefficients() {
				return coefficients;
			}
			
			vector<double> getResiduals() {
				return residuals;
			}
			
			vector<double> getFittedValues() {
				return fitted_values;
			}
			
			vector<double> getEffects() {
				return effects;
			}
			
			vector<vector<double> > getR() {
				return R;
			}
			
			int getRank() {
				return rank;
			}
			
			vector<vector<double> > getQr() {
				return qr;
			}
			
			vector<double> getQraux() {
				return qraux;
			}
			
			vector<int> getPivot() {
				return pivot;
			}
			
			double getTol() {
				return tol;
			}
			
			vector<double> getLinearPredictors() {
				return linear_predictors;
			}
			
			double getDeviance() {
				return deviance;
			}
			
			double getAIC() {
				return aic;
			}
			
			double getNullDeviance() {
				return null_deviance;
			}
			
			int getIter() {
				return iter;
			}
			
			vector<double> getWeights() {
				return weights;
			}
			
			vector<double> getPriorWeights() {
				return prior_weights;
			}
			
			int getDFResidual() {
				return df_residual;
			}
			
			int getDFNull() {
				return df_null;
			}
			
			vector<double> getY() {
				return y;
			}
			
			bool getConverged() {
				return converged;
			}
			
			bool getBoundary() {
				return boundary;
			}
			
			double getTheta() {
				return theta;
			}
			
			double getSETheta() {
				return se_theta;
			}
			
			double getTwoLogLik() {
				return twologlik;
			}
			
			void setCoefficients(vector<double> this_coefficients) {
				coefficients = this_coefficients;
			}
			
			void setResiduals(vector<double> this_residuals) {
				residuals = this_residuals;
			}
			
			void setFittedValues(vector<double> this_fitted_values) {
				fitted_values = this_fitted_values;
			}
			
			void setEffects(vector<double> this_effects) {
				effects = this_effects;
			}
			
			void setR(vector<vector<double> > this_R) {
				R = this_R;
			}
			
			void setRank(int this_rank) {
				rank = this_rank;
			}
			
			void setQr(vector<vector<double> > this_qr) {
				qr = this_qr;
			}
			
			void setQraux(vector<double> this_qraux) {
				qraux = this_qraux;
			}
			
			void setPivot(vector<int> this_pivot) {
				pivot = this_pivot;
			}
			
			void setTol(double this_tol) {
				tol = this_tol;
			}
			
			void setLinearPredictors(vector<double> this_linear_predictors) {
				linear_predictors = this_linear_predictors;
			}
			
			void setDeviance(double this_deviance) {
				deviance = this_deviance;
			}
			
			void setAIC(double this_aic) {
				aic = this_aic;
			}
			
			void setNullDeviance(double this_null_deviance) {
				null_deviance = this_null_deviance;
			}
			
			void setIter(int this_iter) {
				iter = this_iter;
			}
			
			void setWeights(vector<double> this_weights) {
				weights = this_weights;
			}
			
			void setPriorWeights(vector<double> this_prior_weights) {
				prior_weights = this_prior_weights;
			}
			
			void setDfResidual(int this_df_residual) {
				df_residual = this_df_residual;
			}
			
			void setDfNull(int this_df_null) {
				df_null = this_df_null;
			}
			
			void setY(vector<double> this_y) {
				y = this_y;
			}
			
			void setConverged(bool this_converged) {
				converged = this_converged;
			}
			
			void setBoundary(bool this_boundary) {
				boundary = this_boundary;
			}
			
			void setTheta(double this_theta) {
				theta = this_theta;
			}
			
			void setSETheta(double this_se_theta) {
				se_theta = this_se_theta;
			}
			
			void setTwoLogLik(double this_twologlik) {
				twologlik = this_twologlik;
			}
			
			fit(vector<double> &this_coefficients, vector<double> &this_residuals,
							 vector<double> &this_fitted_values, vector<double> &this_effects,
							 vector<vector<double> > &this_R, int this_rank, vector<vector<double> > &this_qr, vector<double> &this_qraux,
							 vector<int> &this_pivot, double this_tol, vector<double> &this_linear_predictors, 
							 double this_deviance, double this_aic, double this_null_deviance, int this_iter, 
							 vector<double> &this_weights, vector<double> &this_prior_weights,
							 int this_df_residual, int this_df_null, vector<double> &this_y, bool this_converged,
							 bool this_boundary) {
				
				coefficients = this_coefficients;
				residuals = this_residuals;
				fitted_values = this_fitted_values;
				effects = this_effects;
				R = this_R;
				rank = this_rank;
				qr = this_qr;
				qraux = this_qraux;
				pivot = this_pivot;
				tol = this_tol;
				linear_predictors = this_linear_predictors;
				deviance = this_deviance;
				aic = this_aic;
				null_deviance = this_null_deviance;
				iter = this_iter;
				weights = this_weights;
				prior_weights = this_prior_weights;
				df_residual = this_df_residual;
				df_null = this_df_null;
				y = this_y;
				converged = this_converged;
				boundary = this_boundary;
			}
			
			fit() {
			}
			
 			~fit() {
 			}
};
