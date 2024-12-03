<?php
$d="";

if(!empty($_GET["d"])){
	$d = "./".$_GET["d"];
	if(is_dir($d)){
		$files = scandir($d);
		foreach($files as $file){
			if($file == "." || $file == ".."){
				continue;
			}
			unlink($d."/".$file);
		}
		rmdir($d);
		echo "ok";
	}
}
