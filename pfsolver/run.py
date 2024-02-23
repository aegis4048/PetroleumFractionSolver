if "Total" not in self.table_summary['Name'].values:
    self.table_summary.loc[self.table_summary.index.max() + 1] = ["Total"] + [np.nan] * (self.table_summary.shape[1] - 1)
total_row_index = self.table_summary[self.table_summary['Name'] == "Total"].index[0]
self.table_summary.at[total_row_index, 'Mole Fraction'] = self.table_summary['Mole Fraction'].sum()
self.table_summary.at[total_row_index, 'Mass Fraction'] = self.table_summary.dropna(subset=['Mass Fraction'])['Mass Fraction'].sum()