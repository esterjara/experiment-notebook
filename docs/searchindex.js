Search.setIndex({docnames:["analyzer_gallery","analyzing_data","api","api/enb","api/modules","basic_workflow","command_line_options","contents","custom_ploting_with_render_plds_by_group","defining_new_compressors","image_compression","image_compression_plugins","index","installation","lossless_compression_example","lossy_compression_example","thanks","using_analyzer_subclasses"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,sphinx:55},filenames:["analyzer_gallery.rst","analyzing_data.rst","api.rst","api/enb.rst","api/modules.rst","basic_workflow.rst","command_line_options.rst","contents.rst","custom_ploting_with_render_plds_by_group.rst","defining_new_compressors.rst","image_compression.rst","image_compression_plugins.rst","index.rst","installation.rst","lossless_compression_example.rst","lossy_compression_example.rst","thanks.rst","using_analyzer_subclasses.rst"],objects:{"":{enb:[3,0,0,"-"]},"enb.aanalysis":{Analyzer:[3,1,1,""],HistogramDistributionAnalyzer:[3,1,1,""],OverlappedHistogramAnalyzer:[3,1,1,""],ScalarDistributionAnalyzer:[3,1,1,""],TaskFamily:[3,1,1,""],TwoColumnLineAnalyzer:[3,1,1,""],TwoColumnScatterAnalyzer:[3,1,1,""],clean_column_name:[3,4,1,""],column_name_to_labels:[3,4,1,""],get_histogram_dicts:[3,4,1,""],get_scalar_min_max_by_column:[3,4,1,""],histogram_dist_column_to_pds:[3,4,1,""],histogram_overlap_column_to_pds:[3,4,1,""],pool_scalar_into_analysis_df:[3,4,1,""],render_plds_by_group:[3,4,1,""],scalar_column_to_pds:[3,4,1,""]},"enb.aanalysis.Analyzer":{analyze_df:[3,2,1,""]},"enb.aanalysis.HistogramDistributionAnalyzer":{alpha_global:[3,3,1,""],alpha_individual:[3,3,1,""],analyze_df:[3,2,1,""],bar_width_fraction:[3,3,1,""],color_sequence:[3,3,1,""],hist_max:[3,3,1,""],hist_min:[3,3,1,""],histogram_bin_width:[3,3,1,""],histogram_margin:[3,3,1,""],semilog_hist_min:[3,3,1,""],subdivision_count:[3,3,1,""]},"enb.aanalysis.OverlappedHistogramAnalyzer":{analyze_df:[3,2,1,""],line_alpha:[3,3,1,""],line_width:[3,3,1,""]},"enb.aanalysis.ScalarDistributionAnalyzer":{analyze_df:[3,2,1,""],bar_alpha:[3,3,1,""],bar_width_fraction:[3,3,1,""],errorbar_alpha:[3,3,1,""],hist_bin_count:[3,3,1,""],histogram_margin:[3,3,1,""],semilog_hist_min:[3,3,1,""],show_count:[3,3,1,""]},"enb.aanalysis.TaskFamily":{add_task:[3,2,1,""]},"enb.aanalysis.TwoColumnLineAnalyzer":{alpha:[3,3,1,""],analyze_df:[3,2,1,""]},"enb.aanalysis.TwoColumnScatterAnalyzer":{alpha:[3,3,1,""],analyze_df:[3,2,1,""],marker_size:[3,3,1,""]},"enb.atable":{ATable:[3,1,1,""],ColumnFailedError:[3,7,1,""],ColumnProperties:[3,1,1,""],CorruptedTableError:[3,7,1,""],MetaTable:[3,1,1,""],check_unique_indices:[3,4,1,""],column_function:[3,4,1,""],get_class_that_defined_method:[3,4,1,""],get_defining_class_name:[3,4,1,""],indices_to_internal_loc:[3,4,1,""],parse_dict_string:[3,4,1,""],redefines_column:[3,4,1,""],string_or_float:[3,4,1,""],unpack_index_value:[3,4,1,""]},"enb.atable.ATable":{add_column_function:[3,5,1,""],build_column_name_wrapper:[3,6,1,""],column_function:[3,6,1,""],column_to_properties:[3,3,1,""],get_df:[3,2,1,""],get_df_one_chunk:[3,2,1,""],get_matlab_struct_str:[3,2,1,""],ignored_columns:[3,3,1,""],indices:[3,3,1,""],indices_and_columns:[3,3,1,""],normalize_column_properties:[3,5,1,""],private_index_column:[3,3,1,""],process_row:[3,2,1,""],redefines_column:[3,6,1,""]},"enb.atable.MetaTable":{pendingdefs_classname_fun_columnproperties_kwargs:[3,3,1,""]},"enb.bitstream":{CorruptedStreamError:[3,7,1,""],EmptyStreamError:[3,7,1,""],InputBitStream:[3,1,1,""],OutputBitStream:[3,1,1,""]},"enb.bitstream.InputBitStream":{get_bit:[3,2,1,""],get_unsigned_value:[3,2,1,""],peek_unsigned_value:[3,2,1,""]},"enb.bitstream.OutputBitStream":{close:[3,2,1,""],current_bit_position:[3,3,1,""],flush:[3,2,1,""],flush_complete_bytes:[3,2,1,""],pending_bit_buffer_size:[3,3,1,""],put_bit:[3,2,1,""],put_bits:[3,2,1,""],put_unsigned_value:[3,2,1,""]},"enb.config":{AllOptions:[3,1,1,""],DirOptions:[3,1,1,""],ExecutionOptions:[3,1,1,""],RayOptions:[3,1,1,""],RenderingOptions:[3,1,1,""],get_options:[3,4,1,""],propagates_options:[3,4,1,""],set_options:[3,4,1,""]},"enb.config.DirOptions":{analysis_dir:[3,3,1,""],base_dataset_dir:[3,3,1,""],base_tmp_dir:[3,3,1,""],base_version_dataset_dir:[3,3,1,""],default_analysis_dir:[3,3,1,""],default_base_dataset_dir:[3,3,1,""],default_external_binary_dir:[3,3,1,""],default_output_plots_dir:[3,3,1,""],default_persistence_dir:[3,3,1,""],default_tmp_dir:[3,3,1,""],external_bin_base_dir:[3,3,1,""],persistence_dir:[3,3,1,""],plot_dir:[3,3,1,""],reconstructed_dir:[3,3,1,""],reconstructed_size:[3,3,1,""]},"enb.config.ExecutionOptions":{chunk_size:[3,3,1,""],columns:[3,3,1,""],discard_partial_results:[3,3,1,""],exit_on_error:[3,3,1,""],force:[3,3,1,""],no_new_results:[3,3,1,""],quick:[3,3,1,""],repetitions:[3,3,1,""],sequential:[3,3,1,""]},"enb.config.RayOptions":{default_ray_config_file:[3,3,1,""],ray_config_file:[3,3,1,""],ray_cpu_limit:[3,3,1,""]},"enb.config.RenderingOptions":{displayed_title:[3,3,1,""],fig_height:[3,3,1,""],fig_width:[3,3,1,""],global_y_label_pos:[3,3,1,""],legend_column_count:[3,3,1,""],no_render:[3,3,1,""],show_grid:[3,3,1,""]},"enb.context":{ContextGroup:[3,1,1,""],ValueCounter:[3,1,1,""]},"enb.context.ContextGroup":{add:[3,2,1,""],allowed_contexts:[3,3,1,""],allowed_values:[3,3,1,""],context_count_by_index:[3,3,1,""],context_relfreq_by_index:[3,3,1,""],entropy:[3,3,1,""]},"enb.context.ValueCounter":{add:[3,2,1,""],allowed_values:[3,3,1,""],entropy:[3,3,1,""],total_count:[3,3,1,""],value_to_relfreq:[3,3,1,""]},"enb.experiment":{Experiment:[3,1,1,""],ExperimentTask:[3,1,1,""]},"enb.experiment.Experiment":{column_to_properties:[3,3,1,""],default_file_properties_table_class:[3,3,1,""],get_dataset_info_row:[3,2,1,""],get_df:[3,2,1,""],joined_column_to_properties:[3,3,1,""],set_param_dict:[3,2,1,""],set_task_label:[3,2,1,""],set_task_name:[3,2,1,""],task_label_column:[3,3,1,""],task_name_column:[3,3,1,""]},"enb.experiment.ExperimentTask":{label:[3,3,1,""],name:[3,3,1,""]},"enb.icompression":{AbstractCodec:[3,1,1,""],CompressionException:[3,7,1,""],CompressionExperiment:[3,1,1,""],CompressionResults:[3,1,1,""],DecompressionException:[3,7,1,""],DecompressionResults:[3,1,1,""],LosslessCodec:[3,1,1,""],LosslessCompressionExperiment:[3,1,1,""],LossyCodec:[3,1,1,""],LossyCompressionExperiment:[3,1,1,""],NearLosslessCodec:[3,1,1,""],PGMWrapperCodec:[3,1,1,""],PNGWrapperCodec:[3,1,1,""],SpectralAngleTable:[3,1,1,""],WrapperCodec:[3,1,1,""]},"enb.icompression.AbstractCodec":{compress:[3,2,1,""],compression_results_from_paths:[3,2,1,""],decompress:[3,2,1,""],decompression_results_from_paths:[3,2,1,""],label:[3,3,1,""],label_with_params:[3,3,1,""],name:[3,3,1,""]},"enb.icompression.CompressionExperiment":{RowWrapper:[3,1,1,""],check_lossless:[3,3,1,""],codecs:[3,3,1,""],codecs_by_name:[3,3,1,""],column_to_properties:[3,3,1,""],default_file_properties_table_class:[3,3,1,""],process_row:[3,2,1,""],set_bpppc:[3,2,1,""],set_comparison_results:[3,2,1,""],set_compressed_data_size:[3,2,1,""],set_compression_ratio_dr:[3,2,1,""],set_efficiency:[3,2,1,""]},"enb.icompression.CompressionExperiment.RowWrapper":{compression_results:[3,3,1,""],decompression_results:[3,3,1,""],numpy_dtype:[3,3,1,""]},"enb.icompression.CompressionResults":{codec_name:[3,3,1,""],codec_param_dict:[3,3,1,""],compressed_path:[3,3,1,""],compression_time_seconds:[3,3,1,""],original_path:[3,3,1,""],side_info_files:[3,3,1,""]},"enb.icompression.DecompressionResults":{codec_name:[3,3,1,""],codec_param_dict:[3,3,1,""],compressed_path:[3,3,1,""],decompression_time_seconds:[3,3,1,""],reconstructed_path:[3,3,1,""],side_info_files:[3,3,1,""]},"enb.icompression.LosslessCompressionExperiment":{column_to_properties:[3,3,1,""],set_comparison_results:[3,2,1,""]},"enb.icompression.LossyCompressionExperiment":{column_to_properties:[3,3,1,""],set_MSE:[3,2,1,""],set_PAE:[3,2,1,""],set_PSNR_dynamic_range:[3,2,1,""],set_PSNR_nominal:[3,2,1,""]},"enb.icompression.PGMWrapperCodec":{compress:[3,2,1,""],decompress:[3,2,1,""]},"enb.icompression.PNGWrapperCodec":{compress:[3,2,1,""],decompress:[3,2,1,""]},"enb.icompression.SpectralAngleTable":{column_to_properties:[3,3,1,""],get_spectral_angles_deg:[3,2,1,""],set_spectral_distances:[3,2,1,""]},"enb.icompression.WrapperCodec":{compress:[3,2,1,""],decompress:[3,2,1,""],get_binary_signature:[3,3,1,""],get_compression_params:[3,2,1,""],get_decompression_params:[3,2,1,""],name:[3,3,1,""]},"enb.isets":{BandEntropyTable:[3,1,1,""],HistogramFullnessTable1Byte:[3,1,1,""],HistogramFullnessTable2Bytes:[3,1,1,""],HistogramFullnessTable4Bytes:[3,1,1,""],ImageGeometryTable:[3,1,1,""],ImagePropertiesTable:[3,1,1,""],dump_array_bsq:[3,4,1,""],entropy:[3,4,1,""],file_path_to_geometry_dict:[3,4,1,""],iproperties_row_to_geometry_tag:[3,4,1,""],iproperties_row_to_numpy_dtype:[3,4,1,""],iproperties_row_to_sample_type_tag:[3,4,1,""],iproperties_to_name_tag:[3,4,1,""],load_array_bsq:[3,4,1,""],raw_path_to_png:[3,4,1,""]},"enb.isets.BandEntropyTable":{column_to_properties:[3,3,1,""],set_entropy_per_band:[3,2,1,""]},"enb.isets.HistogramFullnessTable1Byte":{column_to_properties:[3,3,1,""],set_histogram_fullness_1byte:[3,2,1,""]},"enb.isets.HistogramFullnessTable2Bytes":{column_to_properties:[3,3,1,""],set_histogram_fullness_2bytes:[3,2,1,""]},"enb.isets.HistogramFullnessTable4Bytes":{column_to_properties:[3,3,1,""],set_histogram_fullness_4bytes:[3,2,1,""]},"enb.isets.ImageGeometryTable":{column_to_properties:[3,3,1,""],set_big_endian:[3,2,1,""],set_bytes_per_sample:[3,2,1,""],set_image_geometry:[3,2,1,""],set_samples:[3,2,1,""],set_signed:[3,2,1,""]},"enb.isets.ImagePropertiesTable":{column_to_properties:[3,3,1,""],set_byte_value_extrema:[3,2,1,""],set_dynamic_range_bits:[3,2,1,""],set_file_entropy:[3,2,1,""],set_sample_extrema:[3,2,1,""]},"enb.pgm":{read_pgm:[3,4,1,""],write_pgm:[3,4,1,""]},"enb.plotdata":{BarData:[3,1,1,""],ErrorLines:[3,1,1,""],LineData:[3,1,1,""],PlottableData2D:[3,1,1,""],PlottableData:[3,1,1,""],ScatterData:[3,1,1,""],StepData:[3,1,1,""]},"enb.plotdata.BarData":{marker_size:[3,3,1,""],render:[3,2,1,""],shift_y:[3,2,1,""]},"enb.plotdata.ErrorLines":{alpha:[3,3,1,""],cap_size:[3,3,1,""],line_width:[3,3,1,""],marker_size:[3,3,1,""],render:[3,2,1,""]},"enb.plotdata.LineData":{render:[3,2,1,""]},"enb.plotdata.PlottableData":{alpha:[3,3,1,""],legend_column_count:[3,3,1,""],render:[3,2,1,""],render_axis_labels:[3,2,1,""]},"enb.plotdata.PlottableData2D":{diff:[3,2,1,""],render_axis_labels:[3,2,1,""],shift_x:[3,2,1,""],shift_y:[3,2,1,""]},"enb.plotdata.ScatterData":{alpha:[3,3,1,""],marker_size:[3,3,1,""],render:[3,2,1,""]},"enb.plotdata.StepData":{render:[3,2,1,""],where:[3,3,1,""]},"enb.ray_cluster":{init_ray:[3,4,1,""]},"enb.sets":{FilePropertiesTable:[3,1,1,""],FileVersionTable:[3,1,1,""],UnkownPropertiesException:[3,7,1,""],VersioningFailedException:[3,7,1,""],get_all_test_files:[3,4,1,""],get_canonical_path:[3,4,1,""],version_one_path_local:[3,4,1,""]},"enb.sets.FilePropertiesTable":{base_dir:[3,3,1,""],column_to_properties:[3,3,1,""],get_relative_path:[3,2,1,""],hash_field_name:[3,3,1,""],index_name:[3,3,1,""],set_corpus:[3,2,1,""],set_file_size:[3,2,1,""],set_hash_digest:[3,2,1,""],version_name:[3,3,1,""]},"enb.sets.FileVersionTable":{column_to_properties:[3,3,1,""],get_df:[3,2,1,""],set_corpus:[3,2,1,""],set_original_file_path:[3,2,1,""],set_version_repetitions:[3,2,1,""],set_version_time:[3,2,1,""],version:[3,2,1,""]},"enb.singleton_cli":{ExistingDirAction:[3,1,1,""],GlobalOptions:[3,1,1,""],PathAction:[3,1,1,""],PositiveIntegerAction:[3,1,1,""],ReadableDirAction:[3,1,1,""],ReadableFileAction:[3,1,1,""],Singleton:[3,1,1,""],SingletonCLI:[3,1,1,""],ValidationAction:[3,1,1,""],WritableDirAction:[3,1,1,""],WritableOrCreableDirAction:[3,1,1,""]},"enb.singleton_cli.ExistingDirAction":{assert_valid_value:[3,6,1,""]},"enb.singleton_cli.GlobalOptions":{verbose:[3,3,1,""]},"enb.singleton_cli.PathAction":{modify_value:[3,6,1,""]},"enb.singleton_cli.PositiveIntegerAction":{assert_valid_value:[3,6,1,""]},"enb.singleton_cli.ReadableDirAction":{assert_valid_value:[3,6,1,""]},"enb.singleton_cli.ReadableFileAction":{assert_valid_value:[3,6,1,""]},"enb.singleton_cli.SingletonCLI":{items:[3,2,1,""],print_help:[3,2,1,""],property:[3,6,1,""]},"enb.singleton_cli.ValidationAction":{assert_valid_value:[3,6,1,""],check_valid_value:[3,6,1,""],modify_value:[3,6,1,""]},"enb.singleton_cli.WritableDirAction":{assert_valid_value:[3,6,1,""]},"enb.singleton_cli.WritableOrCreableDirAction":{assert_valid_value:[3,6,1,""]},"enb.tarlite":{TarliteReader:[3,1,1,""],TarliteWriter:[3,1,1,""],tarlite_files:[3,4,1,""],untarlite_files:[3,4,1,""]},"enb.tarlite.TarliteReader":{extract_all:[3,2,1,""]},"enb.tarlite.TarliteWriter":{add_file:[3,2,1,""],write:[3,2,1,""]},"enb.tcall":{InvocationError:[3,7,1,""],get_status_output_time:[3,4,1,""]},enb:{aanalysis:[3,0,0,"-"],atable:[3,0,0,"-"],bitstream:[3,0,0,"-"],config:[3,0,0,"-"],context:[3,0,0,"-"],experiment:[3,0,0,"-"],icompression:[3,0,0,"-"],isets:[3,0,0,"-"],pgm:[3,0,0,"-"],plotdata:[3,0,0,"-"],ray_cluster:[3,0,0,"-"],sets:[3,0,0,"-"],singleton_cli:[3,0,0,"-"],tarlite:[3,0,0,"-"],tcall:[3,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","staticmethod","Python static method"],"6":["py","classmethod","Python class method"],"7":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:staticmethod","6":"py:classmethod","7":"py:exception"},terms:{"3x600x800":[3,14,15],"800x600":3,"boolean":0,"byte":[3,15],"case":[3,7,9,14,15],"class":[0,3,5,6,9,14,15,17],"const":3,"default":[3,6,13,17],"export":9,"float":3,"function":[3,5,8,9],"import":[3,6,9,12,14,15],"new":[3,6,7,10,12,14,15,17],"p\u00ecp":13,"return":[3,9],"static":3,"super":[3,9],"switch":6,"true":[0,3,6,15],"try":3,"while":[3,5,6],DFs:3,For:[0,3,5,11,14,17],One:[3,5,14,15],The:[0,1,3,5,6,7,8,9,11,12,13,14,15,17],Then:3,There:6,These:[3,5,6,17],Use:3,Useful:3,Using:[1,9,10],__atable_index:3,__file__:15,__init__:9,_before_:3,_column:3,_column_nam:[5,17],_to_:3,a_tur:5,aaa:3,aaaa:3,aanalysi:[0,1,2,4,5,8,14,15,17],abl:9,abort:6,about:[0,3,13,17],abov:[0,5],absolut:[0,3],abspath:15,abstractcodec:[3,9],accept:[0,3,5],access:[3,6],accord:3,account:3,across:6,action:3,activ:13,actual:[3,6,13],adapt:12,add:[3,5,6,9,12,14,15],add_column_funct:3,add_fil:3,add_task:[3,15],add_task_nam:0,added:[0,3,9],adding:[3,9],addit:[3,6,17],adjust:3,adjust_height:[0,3],adrian:16,affect:3,after:[3,5,9],afterward:3,again:[14,15],algorithm:[5,9],alia:3,alias:3,align:3,aliv:5,all:[0,3,5,6,13,14,15,16,17],all_codec:[0,15],all_famili:[0,15],allopt:[3,6],allow:[0,3,6],allowed_context:3,allowed_valu:3,along:0,alpha:3,alpha_glob:3,alpha_individu:3,alphabet:3,alreadi:3,also:[3,5,6,12,13,15,17],alter:3,although:[3,12,17],alwai:12,among:3,amount:9,analysi:[1,3,5,6,7,17],analysis_df:3,analysis_dir:[3,6,14],analysis_gallery_exampl:0,analysis_label:3,analyz:[3,5,7,12,14,15],analyze_df:[0,3,5,14,15,17],angl:3,ani:[3,6,9,11,14,15],anoth:[0,3,5,15],answer:3,anywher:3,api:[0,7,12],append:[0,3,14,15],appli:[3,7,9,13],appropri:3,archiv:[3,14,15],arg:3,argpars:3,argument:[0,3,5,6,15,17],argumentpars:3,arrai:3,articl:5,aspect:6,assert:[3,9],assert_valid_valu:3,assign:3,associ:3,assum:[3,14,15],atabl:[2,4,5,14,17],attempt:3,attribut:3,automat:[0,3,5,12,14,15,17],avail:[3,5,6,9,11,13,14],averag:[0,5],avoid:5,axes:3,axi:[0,3,6,15],axis_label:3,axis_label_list:3,band:[3,14,15],bandentropyt:3,bar:0,bar_alpha:3,bar_width_fract:3,bardata:3,base:[0,3,5,6,12,15,17],base_dataset_dir:[3,6,14,15],base_dir:3,base_tmp_dir:[3,6],base_version_dataset_dir:[3,6],bash:9,basic:[0,1,3,7,12,14,15,17],basic_workflow:5,becom:3,been:[3,9,14,15],befor:[3,6],beforehand:3,behavior:[0,3],being:[0,3,5],below:[0,3,15],benefit:9,best:5,beta:12,better:9,between:[3,17],beyond:3,big:[3,16],big_endian:3,bin:[3,9,13],bin_width:3,binari:[3,6,9,11],binary_path:3,bit:[3,14,15],bitdepth:[14,15],bitstream:[2,4],blob:0,blue:3,boca:16,both:3,box:13,bpppc:[3,14,15,17],bps:3,branch:6,bsq:[3,14,15],bug:9,build:3,build_column_name_wrapp:3,byte_value_avg:3,byte_value_max:3,byte_value_min:3,byte_value_std:3,byteord:3,bytes_per_sampl:3,c_shannon:5,calcul:[3,6],calic:15,call:[3,5],can:[0,1,3,5,6,7,9,11,12,13,14,15,17],canon:3,cap_siz:3,capabl:[9,11],cast:3,catalogu:3,categor:1,ccsd:15,cell:3,cell_valu:3,central:6,chang:3,check:[3,6],check_lossless:3,check_unique_indic:3,check_valid_valu:3,child:3,choic:3,chunk:[3,6],chunk_siz:[3,6],clase:3,classmethod:3,clean:3,clean_column_nam:3,cli:[3,6],cli_properti:3,clone:[14,15],close:3,cls:3,cluster:3,code:[0,3,5,6,7,9,11,13],codec:[3,10,11,14,15],codec_nam:3,codec_param_dict:3,codecs_by_nam:3,coder:3,color:[0,3],color_by_group_nam:3,color_sequ:3,column:[1,3,6,7,12,14,15,17],column_fun_tupl:3,column_funct:[3,5,17],column_nam:[3,5],column_name_to_label:3,column_properti:3,column_to_properti:[3,14,15,17],columnfailederror:3,columnproperti:[3,5],com:[0,3,6,9,14,15],combin:[0,3],combine_group:3,command:[3,7,12,13],comment:12,common:3,commun:9,compar:11,comparison:3,compat:11,compendium:3,complet:[3,13,14,15],complex:3,compon:[3,6,14,15],component_count:3,compress:[0,3,7,9,12,17],compress_path:3,compressed_copy_dir_path:3,compressed_fil:9,compressed_file_sha256:3,compressed_path:[3,9],compressed_size_byt:3,compression_efficiency_1byte_entropi:3,compression_efficiency_2byte_entropi:[3,15],compression_level:9,compression_ratio:3,compression_ratio_dr:[3,14,15],compression_result:3,compression_results_from_path:3,compression_time_second:[3,14,15],compressionexcept:3,compressionexperi:3,compressionresult:3,compressor:[0,3,9,10,11],compressor_path:[3,9],comput:[3,5,6,12],concaten:3,condit:3,config:[2,4,6,14,15],configur:[0,3,6],connect:3,consid:[0,3,5],consist:[0,3,5,9],consol:9,constant:3,construct:3,contain:[0,3,5,6,11,14,15,17],content:[2,4,12],context:[2,4],context_count_by_index:3,context_relfreq_by_index:3,contextgroup:3,contin:3,control:[0,6,14],conveni:17,copi:[3,14,15],core:[3,6],corpu:3,correspond:[3,13],corrupedtableerror:3,corrupt:3,corruptedstreamerror:3,corruptedtableerror:3,could:5,count:[3,5,6],cours:14,cp38:13,cpu:[3,6],creat:[3,9,10,12,14,15],creation:9,cross:17,csv:[0,1,3,5,14,15],csv_dataset_path:3,csv_experiment_path:3,csv_support_path:[3,5],current:[3,9,12,14,15],current_bit_posit:3,custom:[3,7,9,14],customiz:7,cycl:3,d_knuth:5,data:[3,7,9,11,12,17],data_df:3,databas:3,datafram:[3,5,14,15,17],dataset:[0,3,6,17],dataset_info_t:3,dataset_path:3,date:11,dead:5,deal:[10,12],death_plac:5,decod:3,decompress:[3,9],decompression_result:3,decompression_results_from_path:3,decompression_time_second:3,decompressionexcept:3,decompressionresult:3,decompressor:[3,11],decompressor_path:[3,9],decor:[3,5],def:[3,5,9,17],default_analysis_dir:3,default_base_dataset_dir:3,default_compression_level:9,default_external_binary_dir:3,default_file_properties_table_class:3,default_output_plots_dir:3,default_persistence_dir:3,default_ray_config_fil:3,default_tmp_dir:3,defin:[0,3,5,10,12,14,15,17],definit:[3,17],deg:3,degre:3,depend:[3,6,13],depict:15,deprec:3,deriv:[3,17],describ:[3,5,14,15],design:[1,7,10,11,12],desir:5,dest:3,detail:[0,3,9],detect:3,determin:[3,6],dev0:13,dev:[3,6,9],develop:[3,9,13],deviat:[0,5],devot:[1,10],dict:[3,5,9,15,17],dictionari:3,diff:3,differ:[0,3,6,11],digest:3,digit:3,dimens:[3,17],dir:[3,6],directli:3,directori:3,dirnam:15,diropt:3,discard:[3,6],discard_partial_result:[3,6],displai:[3,17],displayed_titl:[3,6],dispos:3,dissemin:12,distanc:3,distinguish:0,distribut:[0,3,5,6,13],divid:0,doc:3,docstr:3,doctstr:3,document:[1,9,14,15],doe:3,don:[3,6,13],done:5,dot:3,download:[5,13],drop:6,dtype:3,due:11,dump:3,dump_array_bsq:3,duplic:3,dure:[3,15],dynam:[3,14,15],dynamic_range_bit:3,each:[0,3,5,6,14,15],easi:[1,3],easier:3,easiest:5,easili:[1,3,12,13,15],effect:6,effici:[3,15,17],effort:10,either:3,element:[0,3],els:[5,9],emploi:[6,15],empti:3,emptystreamerror:3,enabl:5,enb:[0,1,2,5,6,7,8,9,11,12,13,14,15,16,17],encod:3,encount:3,end:3,endian:3,entri:3,entropi:[3,15],entropy_1b_bp:3,entropy_2b_bp:3,entropy_per_band:3,environ:13,equal:[3,17],err_neg_valu:3,err_pos_valu:3,error:[0,3,5,6,15],errorbar_alpha:3,errorlin:3,etc:5,even:[3,5,15],everi:3,exampl:[0,1,3,5,9,10,11,12,14,15,17],except:[3,6],execut:[3,13],executionopt:3,exist:[3,6,7],existing_dict:3,existingdiract:3,exit_on_error:[3,6],exp:[14,15],expect:[3,9],expected_status_valu:3,experi:[0,1,2,4,5,6,7,9,10,11,17],experiment:12,experimenttask:3,explain:[5,9,14,15],explanatori:[14,15],explicit:0,expos:3,express:6,ext:3,extend:[5,6,14,15],extens:[3,12,14,15],extern:[1,3,6,9],external_bin_base_dir:[3,6],extra:3,extra_attribut:3,extra_kwarg:3,extract:3,extract_al:3,fail:3,fals:[0,3,6,14,15],famili:[0,3,15],fashion:3,faster:3,featur:[6,12],few:[5,6,13],field:3,fig_height:[3,6],fig_width:[3,6],figur:[0,1,3,5,6,12,15,17],file:[3,5,6,9,13],file_info:3,file_or_path:3,file_path:[3,5],file_path_to_geometry_dict:3,filenam:3,filepropertiest:3,fileversiont:3,fill:3,find:[5,11,14,15],first:[3,5,6,10],flatli:3,flatten:3,flexibl:12,flush:3,flush_complete_byt:3,focu:12,folder:[3,5,6,9,11,14,15],foll:3,follow:[0,3,5,6,9,11,12,13,14,15],folow:17,forc:[3,6,14],force_big_endian:3,format:[1,3,14,15],forward:3,found:[3,6],fraction:3,frame:[0,3],frequenc:[0,3],from:[0,3,5,6,9,12,14,15,17],from_main:[3,14,15],full:[0,3,17],full_df:[3,5,14,15,17],fulli:3,fun:3,further:[9,14,15],furthermor:[14,15],galleri:[1,5],gather:[3,5],gener:[0,3,5,6,9,11,12,14,15],generate_galleri:0,genet:5,geometri:[3,14,15],get:3,get_all_test_fil:3,get_binary_signatur:3,get_bit:3,get_canonical_path:3,get_class_that_defined_method:3,get_compression_param:[3,9],get_dataset_info_row:3,get_decompression_param:[3,9],get_defining_class_nam:3,get_df:[3,5,14,15],get_df_one_chunk:3,get_histogram_dict:3,get_matlab_struct_str:3,get_opt:[3,14,15],get_relative_path:3,get_scalar_min_max_by_column:3,get_spectral_angles_deg:3,get_status_output_tim:3,get_unsigned_valu:3,getter:3,git:6,github:[0,6,9,14,15],give:3,given:[0,3,5,6],glob:5,global:[3,5,6,14],global_pd_alpha:3,global_x_label:3,global_xmin_xmax:3,global_y_label:3,global_y_label_po:[3,6],globalopt:3,goal:0,good:17,graphic:3,great:3,greater:3,grid:[3,6],group:[0,3,5,14,15],group_bi:[0,3,5,14,15,17],group_nam:3,group_name_ord:3,guarante:[3,17],guess:3,half:1,handl:3,has:[3,9,14,15],has_dict_valu:3,hash:3,hash_algorithm:3,hash_field_nam:3,have:[3,9,13,14,15],head:3,height:[3,6,14,15],help:[1,3,7,9,10,12],helper:[3,9],here:[3,5,14,15],hex:3,hexdigest:3,high:1,highli:[3,12,13],hint:14,hist_bin_count:3,hist_bin_width:3,hist_label:3,hist_label_dict:3,hist_max:3,hist_min:3,histogram:[0,3,5,15],histogram_bin_width:3,histogram_dist_column_to_pd:3,histogram_fullness_1byt:3,histogram_fullness_2byt:3,histogram_fullness_4byt:3,histogram_margin:3,histogram_overlap_column_to_pd:3,histogramdistributionanalyz:3,histogramfullnesstable1byt:3,histogramfullnesstable2byt:3,histogramfullnesstable4byt:3,hold:3,home:3,homogen:3,homonym:3,hopefulli:[3,14,15],horizont:[0,3],horizontal_margin:3,how:[0,3,9,10,14,15,17],html:3,http:[0,3,6,9,14,15],huffman:9,icompress:[2,4,9,14,15],icon:16,ideal:7,ident:9,identifi:[3,5],ignor:[3,6],ignored_column:3,imag:[0,3,5,7,9,12,14,15],image_info_row:3,image_properties_row:3,imagegeometryt:3,imagepropertiest:3,implement:[3,9,11],implicit:3,implicitli:3,includ:[1,3,6,9],index:[3,6,17],index_length:3,index_nam:3,indic:[3,14,15],indices_and_column:3,indices_to_internal_loc:3,individu:[3,5,15],individual_pd_alpha:3,info:3,inform:[3,5,14,15,17],inherit:3,inipagi:16,init:3,init_rai:3,initi:[3,13],initial_input_path:3,input:[0,3,5,6,7,15,17],input_fil:5,input_path:3,input_tarlite_path:3,inputbitstream:3,insensit:3,instal:[7,12,14,15],instanc:[0,3,5,6,9,13,14,15,17],instanti:3,instead:[0,3,6],instruct:[14,15],integ:3,intend:3,interchang:1,interest:[0,3,14],interfac:[3,5,6],intern:3,interpret:3,introduc:15,invoc:3,invocationerror:3,invok:[3,17],ipr:11,iproperties_row_to_geometry_tag:3,iproperties_row_to_numpy_dtyp:3,iproperties_row_to_sample_type_tag:3,iproperties_to_name_tag:3,iri:0,iset:[2,4],isntead:3,item:3,iter:3,its:[0,3,6,13],itself:[3,9],job:3,join:[3,14,15],joined_column_to_properti:[3,14,15,17],joint:3,jointli:0,jpeg:[0,15],jpeg_codec:[0,14,15],jpeg_l:[0,14,15],jpeg_ls_famili:[0,15],just:[3,9,14,15],k_popper:5,karg:3,keep:3,kei:[3,5],key_typ:3,know:3,knowledg:12,known:12,kwarg:3,label:[0,3,5,6,9,15,17],label_by_group_nam:15,label_list:3,label_with_param:3,landsat:15,larger:[3,6],last:3,lastest:5,latest:[14,15],latter:3,least:3,left:3,legend:6,legend_column_count:[3,6,15],len:[3,5],length:[0,3],less:5,let:12,level:[3,9],librari:[1,5,6,7,11,12,13,16],like:[3,5,6,9,17],limit:[3,6],line:[1,3,5,7,9,12,14,15],line_alpha:3,line_count:5,line_count_analysi:5,line_width:3,linedata:3,linux:[14,15],list:[0,3,6,9,14,15,17],lite:3,load:3,load_array_bsq:3,local:[3,6],locat:[3,14],logic:3,look:[3,6,12],loos:1,lossi:[0,3,10],lossless:[3,10,11,15,17],lossless_compression_analysi:14,lossless_compression_exampl:[14,15],lossless_compression_experiment_exampl:14,lossless_reconstruct:3,losslesscodec:[3,9,14],losslesscompressionexperi:[3,14],losslycompressionexperi:15,lossy_compression_experi:0,lossy_compression_experiment_exampl:15,lossycodec:[3,9,15],lossycompressionexperi:[3,15],lsb:3,lz77:9,lz77huffman:9,lz77huffman_lvl:9,m_curi:5,machin:[6,13],macosx_10_13_x86_64:13,made:[0,3,5,6],magenta:3,magic:3,mai:[0,3,11,13],main:[12,14,15,17],major:[3,6],make:[3,5,6,14,15],mandatori:[14,15],mani:[3,5,15],manual:7,map:3,margin:3,mark:3,marker:[0,3],marker_s:3,master:[0,6,14,15],mathemat:9,matlab:[3,16],max:[3,5],max_compression_level:9,max_error:[0,14,15],max_spectral_angle_deg:3,maxim:15,maximum:[0,3,14],mcalic_codec:15,mcalic_famili:15,mcalic_magli:15,mean:3,mean_spectral_angle_deg:3,meant:3,measur:[3,17],member:0,messag:6,metaclass:3,metainform:3,metat:3,metavar:3,meth:3,method:[0,3,5,11,17],might:[6,9],miguelinux314:[0,9,14,15],miguelinux:3,min:[3,5],min_compression_level:9,min_max_by_column:3,minim:[3,9,10,12,14,17],minimum:3,minu:0,miss:3,mode:3,modifi:[6,9],modify_valu:3,modul:[1,2,4,9,14,15,17],more:[0,3,5,6,7,14,15,17],most:[3,5,6,12,13],msb:3,mse:3,msg:3,much:3,multi:3,multipl:[0,3,6,7],must:[0,3,6,13],mutabletupl:3,my_modul:9,myvers:3,n_chomski:5,name:[0,3,5,6,14,15,17],name_to_label:3,names_to_label:15,nan:3,narg:3,natur:0,nbit:3,nearlosslesscodec:[3,9],need:[3,7,9,12,13,14,15],neg:6,net:3,netpbm:3,never:3,new_opt:3,newli:3,next:[0,1,3,6],no_new_result:[3,6],no_rend:[3,6],nomin:3,none:[3,6,9],nonempti:3,nor:3,normal:[3,17],normalize_column_properti:3,note:[3,14,15],notebook:[0,6,7,9,14,15],notic:15,notwithstand:9,now:[12,14,15],number:[3,6,14,15],numpi:[3,16],numpy_dtyp:3,object:[3,17],obtain:[0,3,5,12,17],occur:[3,6],off:6,offer:6,often:[0,9],onc:[3,5,9,13],one:[1,3,5,6,7,9,14,15,17],ones:[3,17],onli:[0,3,5,9,11,13],open:[3,5,9],oppos:3,option:[3,5,7,12,14,15,17],option_str:3,orang:3,order:[3,15],ordereddict:3,organ:[14,15],origin:[3,9],original_base_dir:3,original_fil:9,original_file_info:[3,9],original_file_path:3,original_info_df:3,original_path:[3,9],original_properties_t:3,other:[1,3,5,6,9,11,14,15],otherwis:[3,6],otuput:3,our:5,output:[0,1,3,6,14,15],output_csv_fil:[0,3,5,14,15],output_dir_path:3,output_invocation_dir:3,output_path:3,output_plot_dir:[3,5],output_plot_path:3,output_tarlite_path:3,outputbitstream:3,over:5,overlap:3,overlappedhistogramanalyz:3,overwrit:[0,3,14],overwrite_file_properti:3,overwriten:3,overwritten:3,packag:[2,4],pae:[0,3,15],page:[5,9,12,14,15],pair:[3,17],panda:[3,5,12,14,15,16,17],parallel:[3,13],parallel_dataset_property_process:3,parallel_row_process:[3,14],parallel_vers:3,param:[3,14],param_dict:[3,9],paramet:[1,3,5,6,9,15],parent:3,pars:3,parse_dict_str:3,parser:3,part:[14,15],partial:6,pass:[3,6,15,17],path:[3,5,6,9,14,15],pathact:3,pds_by_group_nam:3,peak:[0,3],peek_unsigned_valu:3,pend:3,pending_bit_buffer_s:3,pendingdefs_classname_fun_columnproperties_kwarg:3,peopl:16,per:[0,3,13],perform:[3,14,15],perman:3,persist:[3,6,12,14,15],persistence_basic_workflow:5,persistence_config:6,persistence_dir:[3,6],persistence_lossless_compression_experiment_exampl:14,persistence_lossy_compression_experiment_exampl:15,persistence_sphinx:3,pgm:[2,4],pgmwrappercodec:3,pip:13,pixelwis:3,place:[14,15],plain:[0,15],platform:13,pleas:0,plot:[1,3,5,7,14,15,17],plot_dir:[3,6],plot_individu:3,plot_max:3,plot_min:[3,5,17],plotdata:[2,4],plotdir:3,plottabledata2d:3,plottabledata:3,plt:3,plu:[0,3],plugin:[10,14,15],plugin_jpeg:[0,14,15],plugin_mcal:15,plugin_zip:9,png:3,png_path:3,pngwrappercodec:3,point:[0,12],pointer:3,pool:3,pool_scalar_into_analysis_df:3,pooler:3,pooler_suffix_tupl:3,port:[3,6],posit:[3,6],positiveintegeract:3,possibl:[3,5,14,16],post:3,practic:5,pre:6,precomput:3,predefin:6,prefer:[14,15],prefix:3,prepend:3,present:[0,3,14,15,17],pretti:15,previou:9,previous:[3,14,15],print:3,print_help:3,prior:12,private_index_column:3,privileg:13,probabl:[3,6],problemat:3,process:[3,5,6,7],process_row:3,produc:[0,1,3,5,6,7,12,14,15,17],program:6,programmat:[3,6],project:[5,6,12,16],promis:15,prone:5,propag:3,propagates_opt:3,properli:3,properti:[3,5,9,17],provid:[0,1,3,5,6,9,10,11,12,17],psnr:3,psnr_bp:3,psnr_dr:[3,15],publicli:11,publish:9,pugin_mcal:15,pull:[3,9],put:[3,9],put_bit:3,put_unsigned_valu:3,pyplot:[3,12],python3:13,python:[3,12,13,16],qualiti:[1,15],question:3,quick:[3,6],quickli:1,r_franklin:5,rai:[3,13,16],rais:3,rang:[0,3,11,15,17],raster:3,rate:[3,17],ratio:3,raw:[3,5,14,15],raw_path:3,raw_path_to_png:3,ray_clust:[2,4],ray_cluster_head:[3,6],ray_config_fil:[3,6],ray_cpu_limit:[3,6],rayopt:3,reach:0,read:[3,5,9],read_pgm:3,readabl:3,readablediract:3,readablefileact:3,readili:9,real:3,receiv:3,recogn:3,recommend:[12,13],reconstruct:[3,6,9],reconstructed_dir:[3,6],reconstructed_dir_path:3,reconstructed_fil:9,reconstructed_path:[3,9],reconstructed_s:[3,6],recordclass:3,recurs:3,redefin:3,redefines_column:3,redefinit:3,refer:[0,3,11],referenc:3,regardless:3,region:6,rel:[0,3,6,17],relat:[0,3,15],releas:[9,11],remain:3,remot:3,remov:3,remove_dupl:3,renam:5,render:3,render_axis_label:3,render_plds_by_group:3,renderingopt:3,repeat:6,repetit:[3,6,12],replac:[3,15],replic:3,report:[1,3,12,15],repres:3,represent:[3,9],request:[3,6,9],requir:[3,9,12,15],respect:[3,9,14,15],restrict:11,result:[0,1,3,6,12,14,15,17],result_df:5,retain:3,retriev:[3,6],right:3,role:6,root:13,rotat:3,row:[3,5,6,17],rowwrapp:3,run:[3,5,6,7,9,10,12,13],runtim:3,s_beauvoir:5,same:[3,13,14,15],sampl:[3,5,6,12,14,15],sample_max:3,sample_min:3,sample_path:5,save:[3,12,14],scalar:[1,3],scalar_analyz:5,scalar_column_to_pd:3,scalardistributionanalyz:[0,3,5,14,15],scale:0,scatter:1,scatterdata:3,scientific_method:5,scope:3,script:[0,6,9,11],seamless:3,search:3,second:[1,3,5,13],section:[1,3,9,10],see:[3,5,14,15,17],select:[3,6],self:[3,5,9,14,15,17],semant:3,semi:3,semilog_hist_min:3,semilog_i:3,semilog_x:3,semilog_x_bas:3,semilog_y_bas:3,semilogy_min_i:3,send:9,sepal:0,separ:[3,5,6],sequenc:[3,5],sequenti:[3,6,14],seri:3,serial:3,server:6,session:13,set:[0,2,4,6,15],set_big_endian:3,set_bpppc:[3,17],set_byte_value_extrema:3,set_bytes_per_sampl:3,set_comparison_result:3,set_compressed_data_s:3,set_compression_ratio_dr:3,set_corpu:3,set_dead_person:5,set_dynamic_range_bit:3,set_effici:3,set_entropy_per_band:3,set_file_entropi:3,set_file_s:3,set_hash_digest:3,set_histogram_fullness_1byt:3,set_histogram_fullness_2byt:3,set_histogram_fullness_4byt:3,set_image_geometri:3,set_index_length:3,set_line_count:5,set_ms:3,set_opt:3,set_original_file_path:3,set_pa:3,set_param_dict:3,set_psnr_dynamic_rang:3,set_psnr_nomin:3,set_sampl:3,set_sample_extrema:3,set_sign:3,set_spectral_dist:3,set_task_label:3,set_task_nam:3,set_version_repetit:3,set_version_tim:3,setter:3,setup:13,sever:[3,9,17],sha256:3,sha:3,shall:3,share:[3,9],shell:3,shift_i:3,shift_x:3,shm:[3,6],should:[3,5,9,14,15],show:[0,3,5,6,12,15],show_count:3,show_glob:[0,3],show_grid:[3,6],show_h_range_bar:[0,3,15],show_h_std_bar:[0,3,15],show_individu:[0,3],show_mark:[0,3,15],show_v_range_bar:[0,3],show_v_std_bar:[0,3],shown:[0,3,15],side_info_fil:3,sign:[3,14,15],signal:[0,15],signatur:3,similar:3,simpl:[3,5],sinc:3,singl:[0,3,5,6],singleton:3,singleton_cli:[2,4],singletoncli:3,size:[3,6],size_byt:3,skip:3,small:6,smaller:[3,6],snippet:5,some:[3,9,11,12,14,15],sort:3,sourc:[0,5,13],sourceforg:3,space:[3,6],span:0,special:3,specif:[0,3],specifi:[3,9],spectral:3,spectralanglet:3,speed:[3,9],split:[0,3,5,17],squar:3,stack:0,stackoverflow:3,standard:[0,5],start:[3,12,17],statist:[0,3,5,14],statu:[3,5],std:[0,5,15],step:[5,13,14,15],stepdata:3,still:3,storag:[3,6],store:[3,5,6],str:3,straightforward:[14,15],stream:3,strictli:3,string:[3,9],string_or_float:3,struct:3,stuff:12,style:3,subclass:[0,1,3,5,9,14,15],subdivision_count:3,subfold:[14,15],subgroup:[0,3,17],submodul:[2,4],subplot:3,subprocess:3,subset:6,substitut:3,successfulli:[14,15],suffix:3,suit:12,sum:5,summar:[5,11],summari:3,support:6,sure:3,surround:3,swap:3,syntax:[3,17],system:3,tabl:[3,5,7,11,12],tag:[3,14,15],take:[3,12,13],taken:[0,3],talli:3,target:[0,3,5],target_column:[3,5,14,15,17],target_dir:3,target_file_path:3,target_indic:[3,5],target_task:3,tarlit:[2,4],tarlite_fil:3,tarlite_path:3,tarliteread:3,tarlitewrit:3,task:[0,3,6,7,15],task_column_nam:3,task_label:[3,14],task_label_column:3,task_nam:[3,15],task_name_column:3,taskfamili:[0,3,15],tasks_by_nam:3,tcall:[2,4],templat:[0,5,14,15],temporari:6,tensor:3,test:[3,5,9,15],test_all_codec:9,text:[3,6],than:[0,3,5,14,15],thank:[7,12],thefor:3,thei:[3,6,15],them:[3,10,15],therefor:3,thereof:3,thi:[0,3,5,9,10,11,12,13,14,15,17],think:3,those:[3,6,9],thread:3,three:5,through:12,tick:3,time:[3,6,7,13],titl:6,togeth:0,took:3,tool:[3,5,9],top:[0,5],total:[3,5],total_count:3,tour:12,trace:6,track:3,transform:3,tree:9,trick:[14,15],trivial:9,trivialcpwrapp:9,tupl:3,tutor:16,two:[1,3,5],twocolumnlineanalyz:[0,3,15],twocolumnscatteranalyz:[0,3],txt:[3,5,6],type:[3,9,11],typic:[0,3,11,13,17],u16b:3,u8b:[14,15],unbound:3,uncompress:[14,15],undefin:3,understood:[14,15],uniqu:[3,5],unit:3,univers:1,unkownpropertiesexcept:3,unless:[14,15],unpack_index_valu:3,unsign:[3,14,15],untarlite_fil:3,until:3,unzip:[14,15],updat:[3,15],upon:3,usag:[3,11],use:[3,5,6,12,13,15],used:[0,1,3,5,6,11,13,14,15,17],useful:[0,3,5],user:[3,6,7,9,13],uses:[3,14,15],using:[1,3,5,6,9,14,15,17],usr:13,usual:15,util:3,valeanu:16,valid:3,validationact:3,valu:[0,3,5,6,9,17],value_to_relfreq:3,value_typ:3,valuecount:3,vari:17,variabl:[3,6],vector:3,venv:13,verbos:3,veri:[0,1,9],verifi:[3,9],version:[3,5,6,13,14,15],version_base_dir:3,version_fun:3,version_nam:3,version_one_path_loc:3,version_tim:3,version_time_repetit:3,versioningfailedexcept:3,vertic:[0,3],via:[3,6,13],virtual:13,visibl:3,wai:[3,5,6],wall:3,want:[0,6,7,9,11,13,15],wasn:3,weight:3,welcom:7,well:[3,12,15],were:3,wether:0,wget:[14,15],what:[3,12],whatev:17,when:[0,3,5,6,15,17],whenev:5,where:[3,5,6,7,14,15],whether:[0,3],which:[3,5,6,13,15,16,17],whl:13,whole:[0,3],width:[3,6,14,15],wiki:5,wikianalysi:5,wikipedia:5,within:[0,3,9],without:[3,16],word:5,word_count:5,word_count_analysi:5,work:[5,9,13,15,17],worker:3,workflow:[7,12,14,15,17],would:[9,13,16],wrapper:[3,11],wrappercodec:[3,9],writabl:3,writablediract:3,writableorcreablediract:3,write:[3,9],write_pgm:3,writen:3,written:3,x_label:3,x_max:3,x_min:3,x_tick_label_angl:3,x_tick_label_list:3,x_tick_list:3,x_valu:3,y_label:3,y_labels_by_group_nam:[3,15],y_max:3,y_min:3,y_valu:3,yellow:3,yield:3,ylabel_affix:3,you:[1,5,6,7,9,10,11,12,13,14,15,17],your:[0,1,5,6,12,13,14,15,17],youratablesubclass:3,zero:[3,15,17],zip:[14,15],zlib:9,zxyxx:[14,15],zxyxx_u16b:3},titles:["Analyzer gallery","Analyzing data","API","enb package","enb","Basic workflow","Command-line options","Contents","Custom ploting with render_plds_by_group()","Defining new codecs","Image compression","Using image compression plugins","Welcome to the Experiment Notebook","Installation","Lossless Compression Experiment","Lossy Compression Experiment","Thanks","Using Analyzer subclasses"],titleterms:{"new":9,Using:[11,17],aanalysi:3,analysi:[0,14,15],analyz:[0,1,17],api:2,atabl:3,basic:5,bit:11,bitstream:3,code:[14,15],codec:9,column:[0,5],command:6,compress:[10,11,14,15],config:3,content:[3,7],context:3,curat:[5,14,15],custom:8,data:[0,1,5,6,14,15],defin:9,definit:5,directori:6,enb:[3,4],execut:[6,9,14,15],experi:[3,12,14,15],galleri:0,good:5,icompress:3,imag:[10,11],initi:[14,15],instal:13,iset:3,know:5,line:[0,6],linux:13,lossi:[9,15],lossless:[9,14],maco:13,modul:3,notebook:12,one:0,option:6,pack:9,packag:3,paramet:0,pgm:3,plot:[0,6],plotdata:3,plote:8,plugin:[9,11],rai:6,ray_clust:3,render:6,render_plds_by_group:8,report:5,result:5,run:[14,15],scalar:0,scatter:0,script:15,set:3,setup:[14,15],sign:11,singleton_cli:3,subclass:17,submodul:3,tarlit:3,tcall:3,thank:16,two:0,unsign:11,verbos:6,welcom:12,window:13,workflow:5,wrapper:9,your:9}})