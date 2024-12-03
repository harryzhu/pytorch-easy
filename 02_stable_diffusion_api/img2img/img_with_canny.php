<?php
require_once("p_np.php");


//$p = 'best quality, photorealistic, 8k, high res, full color, ((a woman with long hair)), (((in a white shirt and pants))),( is dancing and singing into a microphone, in a crowd, while other pepole are watcheing this woman, in a street),<lora:nashiko:0.65> <lora:tsuchiyatao_lora:0.25>';









function getImageBase64($fpath){
	$img = base64_encode(file_get_contents($fpath));
	return $img;
}

function getBlip($img_name){
	//$bfile = "data/img2blip/".str_replace(".png",".txt",$img_name);
	$bfile = "data/img2blip/_all_blip.txt";
	$blip_text = file_get_contents($bfile);
	if(trim($blip_text) == ""){
		print("blip text is empty: ".$bfile);
	}
	return $blip_text;
}



function getPayload($img_name,$model_name,$seed_fixed,$p,$np){
	$scripts = array();

	$scripts_controlnet = array();
	$scripts_controlnet["args"] = array();
	$scripts_controlnet_args = array(
		"enabled" => true,
		"input_image" => "data:image/png;base64,".getImageBase64("data/img_src/".$img_name),//"data:image/png;base64,".
		"module" => "canny",
		"model" => "control_v11p_sd15_canny [d14c016b]",
		"weight" =>  1.5,
		"resize_mode" =>  1,
		"processor_res" =>  1024,
		"pixel_perfect" => false, 
		"threshold_a" =>100,
		"threshold_b" =>200,
		//"control_mode" => 2,
		"control_mode" => 'My prompt is more important'
	);
	array_push($scripts_controlnet["args"], $scripts_controlnet_args)  ;

	$scripts["ControlNet"] = $scripts_controlnet;

	$payload = array();

	//$payload["prompt"]= '(('.getBlip($img_name).")),".$p;
	$payload["prompt"]= $p;
	$payload["negative_prompt"]= $np;
	$payload["override_settings"]= array(
		"sd_model_checkpoint"=> $model_name,
	);
	$payload["model"]=4;# inpaint upload mask
	$payload["seed"]= $seed_fixed;
	$payload["batch_size"]= 2;
	$payload["n_iter"]= 1;
	$payload["steps"]= 20;
	$payload["cfg_scale"]= 7;
	$payload["width"]= 576;
	$payload["height"]= 1024;
	$payload["restore_faces"]= false;
	$payload["tiling"]= false;
	$payload["eta"]= 0;
	$payload["script_args"]= [];
	$payload["sampler_index"]= "DPM++ 2M Karras";
	//$payload["sampler_index"]= "Euler a";
	//$payload["init_images"]= [getImageBase64("data/img2canny/".$img_name)];
$payload["init_images"]= [getImageBase64("data/img_src/".$img_name)];
$payload["init_img_inpaint"]=0 ;
$payload["init_mask_inpaint"]= getImageBase64("data/img2canny/".$img_name);
	$payload["resize_mode"]= 1;
	$payload["denoising_strength"]= 0.7;
	$payload["mask_blur"]=4;
	$payload["inpainting_fill"]=1;
	$payload["inpaint_full_res"]= true;
	$payload["inpaint_full_res_padding"]= 32;
	$payload["inpainting_mask_invert"]= 1;
//$payload["alwayson_scripts"]= [];//$scripts_controlnet;
	$payload["alwayson_scripts"]= $scripts;

	//print(json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
	return $payload;
}



?>
<?php

function request_by_curl($remote_server, $post_data) {
	$ch = curl_init();
	curl_setopt($ch, CURLOPT_URL, $remote_server);
	curl_setopt($ch, CURLOPT_HTTPHEADER,  array("Content-Type: application/json"));
	curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
	curl_setopt($ch, CURLOPT_POST, 1);
	curl_setopt($ch, CURLOPT_POSTFIELDS, $post_data);
	$data = curl_exec($ch);
	curl_close($ch);

	return $data;
}

// $opt = file_get_contents('http://172.16.10.119:7860/sdapi/v1/options');
// $arr_opt = json_decode($opt,true);
// print_r($arr_opt);
// $arr_opt["sd_model_checkpoint"] = $model_name;
// request_by_curl('http://172.16.10.119:7860/sdapi/v1/options',json_encode($arr_opt));

function post_sd_api($payload,$img_name){
	$res = request_by_curl("http://192.168.0.108:7860/sdapi/v1/img2img",json_encode($payload));
	$arr_res = json_decode($res,true);
	echo "<hr/>";
	print_r($img_name);
	//echo "<hr/>";
	$arr_res_images = $arr_res["images"];
	if(sizeof($arr_res_images)>1){
		array_pop($arr_res_images);
	}

	foreach ($arr_res_images as $k=>$v) {
		$img_dir = "data/img_dst_canny/".$img_name;
		if(!is_dir($img_dir)){
			mkdir($img_dir,0777);
		}

		file_put_contents($img_dir."/".str_replace(".png", "_".$k.".png", $img_name), base64_decode($v));
 	//print("<img src='/test/img2img/".$img_dir."/".$img_name."' />");
		//print("<img src='data:image/png;base64,".$v."' />");
	} 
}

$img_list = scandir("data/img2canny");
sort($img_list);
#$img_list = array_slice($img_list,11);
$max_size = 3;
$idx = 1;
foreach ($img_list as $img) {
	if($idx > $max_size){
		break;
	}
	if(substr($img, -4) != ".png"){
		continue;
	}
	$img_dir = "data/img_dst_canny/".$img;
	if(is_dir($img_dir)){
		echo "<br/>skip: ".$img;
		$idx += 1;
		continue;
	}

	$payload = getPayload($img,$model_name,$seed_fixed,$p,$np);
	post_sd_api($payload,$img);
	$idx += 1;
}

// $img = $img_list[2];
// $img = "img-00001.png";
// echo $img;
// echo "<hr/>";
// $payload = getPayload($img,$model_name,$seed_fixed,$p,$np);
// print_r($payload);
// echo "<hr/>";
// post_sd_api($payload,$img);


#$payload = getPayload($img_name,$model_name,$seed_fixed,$p,$np);
#post_sd_api($payload,$img_name);

//print("<img src='data:image/png;base64,".$arr_res["images"][0]."' />");

?>


