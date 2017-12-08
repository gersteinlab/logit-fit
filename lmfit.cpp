/* A data structure for holding the return values of the Cdqrls function */
class lmfit {
	private: // Member variables
		vector<vector<double> > qr;
		vector<double> coefficients;
		vector<double> residuals;
		vector<double> effects;
		int rank;
		vector<int> pivot;
		vector<double> qraux;
		double tol;
		int pivoted;
	
	public: // Set/get methods
		vector<vector<double> > getQr() {
			return qr;
		}
		
		vector<double> getCoefficients() {
			return coefficients;
		}
		
		vector<double> getResiduals() {
			return residuals;
		}
		
		vector<double> getEffects() {
			return effects;
		}
		
		int getRank() {
			return rank;
		}
		
		vector<int> getPivot() {
			return pivot;
		}
		
		vector<double> getQraux() {
			return qraux;
		}
		
		double getTol() {
			return tol;
		}
		
		int getPivoted() {
			return pivoted;
		}
		
		void setQr(vector<vector<double> > &this_qr) {
			qr = this_qr;
		}
		
		void setCoefficients(vector<double> &this_coefficients) {
			coefficients = this_coefficients;
		}
		
		void setResiduals(vector<double> &this_residuals) {
			residuals = this_residuals;
		}
		
		void setEffects(vector<double> &this_effects) {
			effects = this_effects;
		}
		
		void setRank(int this_rank) {
			rank = this_rank;
		}
		
		void setPivot(vector<int> &this_pivot) {
			pivot = this_pivot;
		}
		
		void setQraux(vector<double> &this_qraux) {
			qraux = this_qraux;
		}
		
		void setTol(double this_tol) {
			tol = this_tol;
		}
		
		void setPivoted(int this_pivoted) {
			pivoted = this_pivoted;
		}
		
		lmfit(vector<vector<double> > &this_qr, vector<double> &this_coefficients, 
					vector<double> &this_residuals, vector<double> &this_effects, 
					int this_rank, vector<int> &this_pivot, vector<double> &this_qraux, double this_tol, 
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
