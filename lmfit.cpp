/* A data structure for holding the return values of the Cdqrls function */
class lmfit {
	private: // Member variables
		double** qr;
		double* coefficients;
		double* residuals;
		double* effects;
		int rank;
		int* pivot;
		double* qraux;
		double tol;
		int pivoted;
	
	public: // Set/get methods
		double** getQr() {
			return qr;
		}
		
		double* getCoefficients() {
			return coefficients;
		}
		
		double* getResiduals() {
			return residuals;
		}
		
		double* getEffects() {
			return effects;
		}
		
		int getRank() {
			return rank;
		}
		
		int* getPivot() {
			return pivot;
		}
		
		double* getQraux() {
			return qraux;
		}
		
		double getTol() {
			return tol;
		}
		
		int getPivoted() {
			return pivoted;
		}
		
		void setQr(double** this_qr) {
			qr = this_qr;
		}
		
		void setCoefficients(double* this_coefficients) {
			coefficients = this_coefficients;
		}
		
		void setResiduals(double* this_residuals) {
			residuals = this_residuals;
		}
		
		void setEffects(double* this_effects) {
			effects = this_effects;
		}
		
		void setRank(int this_rank) {
			rank = this_rank;
		}
		
		void setPivot(int* this_pivot) {
			pivot = this_pivot;
		}
		
		void setQraux(double* this_qraux) {
			qraux = this_qraux;
		}
		
		void setTol(double this_tol) {
			tol = this_tol;
		}
		
		void setPivoted(int this_pivoted) {
			pivoted = this_pivoted;
		}
		
		lmfit(double** this_qr, double* this_coefficients, 
					double* this_residuals, double* this_effects, 
					int this_rank, int* this_pivot, double* this_qraux, double this_tol, 
					int this_pivoted) {
							 
			qr = this_qr;
			coefficients = this_coefficients;
			residuals = this_residuals;
			effects = this_effects;
			rank = this_rank;
			pivot = this_pivot;
			qraux = this_qraux;
			tol = this_tol;
			pivoted = this_pivoted;
		}
		
		lmfit() {
		}
		
		~lmfit() {
		}
};
